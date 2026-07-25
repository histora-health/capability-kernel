"""The token trie: what the model may emit next, as a set of token ids.

Ported in spirit from EvolvingAgentsLabs/token-trie and narrowed to what this
kernel needs. Two things are different here, and both come from the domain.

**Slots are first-class.** A rename takes a name nobody enumerated. The trie
holds a node that accepts any token whose text does not close the argument,
bounded by a cap, and resumes the fixed grammar on the closing token. Inside a
slot the grammar constrains *shape*, not content — which is exactly what GBNF
gives and no more, and is worth saying out loud rather than implying otherwise.

**The trie is rebuilt from state, not loaded once.** A file that stops existing
stops being emittable, so the compiler runs whenever the world moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotSpec:
    """A region whose content cannot be enumerated.

    Membership depends on what a token *says*, which means detokenising it —
    so the trie cannot decide it alone and hands the question to the caller.
    """

    #: Decoded text matching this may not appear as content.
    forbid: str
    #: Hard cap. At the cap only the exit remains legal.
    max_tokens: int = 24
    #: Below this the slot cannot close.
    min_tokens: int = 1
    name: str = "slot"
    exit_tokens: set[int] = field(default_factory=set)

    def allows(self, text: str, consumed: int) -> bool:
        if consumed >= self.max_tokens:
            return False
        return not any(ch in text for ch in self.forbid)

    def may_exit(self, consumed: int) -> bool:
        return consumed >= self.min_tokens


@dataclass
class SlotState:
    """Returned instead of a set when generation is inside a slot."""

    spec: SlotSpec
    consumed: int

    @property
    def exit_tokens(self) -> set[int]:
        return self.spec.exit_tokens

    def allows(self, text: str) -> bool:
        return self.spec.allows(text, self.consumed)

    @property
    def may_exit(self) -> bool:
        return self.spec.may_exit(self.consumed)

    @property
    def fallback(self) -> int:
        """Closing the argument is the only choice guaranteed to stay legal.
        Content would be a guess."""
        return next(iter(self.spec.exit_tokens), -1)


class _Node:
    __slots__ = ("children", "is_end", "opcode", "slot")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.is_end = False
        self.opcode = -1
        self.slot: SlotSpec | None = None


class TokenTrie:
    """Legal token sequences, keyed by token id."""

    def __init__(self) -> None:
        self.root = _Node()
        self.opcodes: list[tuple[str, str]] = []  # (text, label)

    def __len__(self) -> int:
        return len(self.opcodes)

    def insert(self, text: str, tokens: list[int], label: str = "") -> int:
        idx = len(self.opcodes)
        node = self.root
        for tok in tokens:
            node = node.children.setdefault(tok, _Node())
        node.is_end = True
        node.opcode = idx
        self.opcodes.append((text, label))
        return idx

    def insert_with_slot(self, text: str, prefix: list[int], spec: SlotSpec,
                         suffix: list[int], label: str = "") -> int:
        """Insert ``prefix`` + a free region + ``suffix``.

        The first token of the suffix is what closes the slot, so leaving the
        free region and resuming the fixed grammar are the same operation.
        """
        if not prefix or not suffix:
            raise ValueError("a slot needs a prefix to open it and a suffix to close it")

        idx = len(self.opcodes)
        node = self.root
        for tok in prefix:
            node = node.children.setdefault(tok, _Node())

        node.slot = spec
        spec.exit_tokens.add(suffix[0])

        for tok in suffix:
            node = node.children.setdefault(tok, _Node())
        node.is_end = True
        node.opcode = idx
        self.opcodes.append((text, label))
        return idx

    # ── Walking ──────────────────────────────────────────────────────────────

    def _walk(self, tokens: list[int]):
        node, slot_node, consumed = self.root, None, 0
        for tok in tokens:
            if slot_node is not None:
                if tok in slot_node.slot.exit_tokens:
                    nxt = node.children.get(tok)
                    if nxt is None:
                        return None, None, 0
                    node, slot_node, consumed = nxt, None, 0
                else:
                    consumed += 1
                continue

            nxt = node.children.get(tok)
            if nxt is None:
                return None, None, 0
            node = nxt
            if node.slot is not None:
                slot_node, consumed = node, 0
        return node, slot_node, consumed

    def next_tokens(self, tokens: list[int]) -> set[int] | SlotState | None:
        """Legal continuations. ``None`` means the prefix left the trie."""
        node, slot_node, consumed = self._walk(tokens)
        if node is None:
            return None
        if slot_node is not None:
            return SlotState(slot_node.slot, consumed)
        return set(node.children.keys())

    def is_complete(self, tokens: list[int]) -> bool:
        node, slot_node, _ = self._walk(tokens)
        # An opcode is not complete while a slot is open, however much text has
        # accumulated: the closing token has not been emitted.
        return node is not None and slot_node is None and node.is_end

    def opcode_at(self, tokens: list[int]) -> tuple[str, str] | None:
        node, slot_node, _ = self._walk(tokens)
        if node is None or slot_node is not None or not node.is_end:
            return None
        return self.opcodes[node.opcode]
