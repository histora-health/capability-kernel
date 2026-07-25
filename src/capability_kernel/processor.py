"""The mask, applied inside the decoding loop.

This is the file the whole argument rests on. Everything upstream describes a
capability surface; this is where the description stops being advisory. A token
outside the surface does not get generated and rejected — it is set to negative
infinity before the sampler sees it, so it has no probability of being chosen.

**Prose stays free.** Masking every token would make the model unable to say
"I cannot do that", which is a real and necessary answer. So the processor
watches for the frame and only takes over once the model has committed to
acting: free until ``\\nACTION``, enforced from there to the end of the action,
free again afterwards. Gemma4's mandatory thought channel passes through the
free segment untouched, which is what makes this model usable at all.

**The pressure is recorded before it is discarded.** At every enforced step the
processor sums the probability the model assigned to tokens the mask forbids.
That number does not survive normal sampling — it is destroyed by the mask that
makes it harmless — and it is the one signal that says what the model *wanted*
to do. A log tells you what happened; this tells you what was prevented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compiler import CompiledSurface
from .trie import SlotState

NEG_INF = -float("inf")


@dataclass
class Step:
    """One enforced decoding step."""

    #: How many tokens the mask left standing, out of the whole vocabulary.
    allowed: int
    #: Probability the model put on tokens the mask forbids, before masking.
    #: Zero means the model was already going to comply. High means the mask
    #: is the only reason it did.
    rejected_mass: float
    #: The token actually chosen, and whether it was the model's own argmax.
    chosen: int = -1
    was_argmax: bool = True
    in_slot: str | None = None


@dataclass
class Telemetry:
    """What the mask prevented, per action.

    Reported rather than merely collected. A mask that never rejects anything is
    indistinguishable from no mask at all, and worth knowing about: it means the
    measurement is not testing what it claims to.
    """

    steps: list[Step] = field(default_factory=list)
    actions: int = 0

    @property
    def enforced_steps(self) -> int:
        return len(self.steps)

    @property
    def peak_pressure(self) -> float:
        return max((s.rejected_mass for s in self.steps), default=0.0)

    @property
    def mean_pressure(self) -> float:
        return sum(s.rejected_mass for s in self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def diverted_steps(self) -> int:
        """Steps where the model's own top choice was forbidden.

        The count that matters: each one is a token that would have been
        generated, and in the baseline arm would have had to be caught
        afterwards — or not caught.
        """
        return sum(1 for s in self.steps if not s.was_argmax)

    def summary(self) -> dict:
        return {
            "actions": self.actions,
            "enforced_steps": self.enforced_steps,
            "diverted_steps": self.diverted_steps,
            "peak_pressure": round(self.peak_pressure, 4),
            "mean_pressure": round(self.mean_pressure, 4),
            "narrowest": min((s.allowed for s in self.steps), default=0),
        }


class CapabilityProcessor:
    """A llama.cpp ``LogitsProcessor`` that enforces the compiled surface.

    :param surface: the compiled trie, rebuilt whenever the world moves.
    :param open_tokens: the token sequence that hands control to the mask.
    :param detokenize: needed for slots — whether a token is legal inside a free
        argument depends on what it *says*, which only the tokenizer knows.
    :param enabled: opcode indices the phase controller currently permits. None
        means all of them. This is what makes the surface a function of state
        rather than a fixed grammar: a phase that forbids moving does not
        validate moves and reject them, it compiles a trie without them.
    """

    def __init__(self, surface: CompiledSurface, open_tokens: list[int],
                 detokenize, enabled: set[int] | None = None,
                 telemetry: Telemetry | None = None) -> None:
        self.surface = surface
        self.open_tokens = list(open_tokens)
        self.detokenize = detokenize
        self.enabled = enabled if enabled is not None else surface.all_indices
        self.telemetry = telemetry if telemetry is not None else Telemetry()

        self._path: list[int] = []      # tokens consumed inside the current action
        self._prompt_len: int | None = None

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return bool(self._path)

    def reset(self) -> None:
        self._path, self._prompt_len = [], None

    def _generated(self, input_ids) -> list[int]:
        if self._prompt_len is None:
            self._prompt_len = len(input_ids)
        return list(input_ids[self._prompt_len:])

    def _locate(self, gen: list[int]) -> list[int]:
        """Where the current action starts, derived from what was generated.

        Deliberately not tracked incrementally. llama.cpp does not tell a
        processor which token the sampler chose, so a hand-maintained path
        needs the caller to feed every token back — two sources of truth that
        drift the moment anything retries, rewinds or batches. ``input_ids``
        already carries the answer.
        """
        n = len(self.open_tokens)
        for i in range(len(gen) - n, -1, -1):
            if gen[i:i + n] == self.open_tokens:
                path = gen[i:]
                # A finished action releases the mask; prose after it is free
                # until the model spells the frame again.
                return [] if self.surface.trie.is_complete(path) else path
        return []

    # ── The mask ─────────────────────────────────────────────────────────────

    def __call__(self, input_ids, scores):
        gen = self._generated(input_ids)
        was_active = bool(self._path)
        self._path = self._locate(gen)

        if not self._path:
            # Free prose — including gemma4's thought channel, which must pass
            # through untouched or the model cannot think before acting.
            if was_active:
                self.telemetry.actions += 1
            return scores

        allowed = self._allowed_now(scores)
        return self._apply(scores, allowed)

    def _allowed_now(self, scores) -> set[int]:
        """Legal continuations, restricted to the enabled opcodes."""
        nxt = self.surface.trie.next_tokens(self._path)

        if nxt is None:
            # Should be unreachable: every token in _path came through the mask.
            # If it happens, the trie and the sampler have desynchronised, and
            # continuing would silently produce unconstrained output.
            raise RuntimeError(
                f"the walk left the trie at {self._path!r} — tokenizer parity "
                f"has broken; re-run parity_report before trusting this run"
            )

        if isinstance(nxt, SlotState):
            return self._slot_tokens(nxt, scores)

        return {t for t in nxt if self._reaches_enabled(t)}

    def _reaches_enabled(self, token: int) -> bool:
        """Prune branches leading only to opcodes this phase forbids."""
        if self.enabled == self.surface.all_indices:
            return True
        return bool(self._opcodes_under(self._path + [token]) & self.enabled)

    def _opcodes_under(self, path: list[int]) -> set[int]:
        node, slot, _ = self.surface.trie._walk(path)
        if node is None:
            return set()
        out: set[int] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_end:
                out.add(n.opcode)
            stack.extend(n.children.values())
        return out

    def _slot_tokens(self, state: SlotState, scores) -> set[int]:
        """Which tokens may appear inside a free argument.

        Decided by decoding candidates, so this is the expensive step. Only the
        top slice is considered: the rest carry so little probability that
        admitting them changes nothing, and decoding the whole vocabulary at
        every slot token would dominate the run.
        """
        allowed: set[int] = set(state.exit_tokens) if state.may_exit else set()

        for token in np.argsort(scores)[::-1][:256]:
            token = int(token)
            if token in allowed:
                continue
            try:
                text = self.detokenize([token])
            except Exception:
                continue
            if state.allows(text):
                allowed.add(token)

        # A slot with nothing legal left must still be able to close, or
        # generation stalls at negative infinity everywhere.
        return allowed or set(state.exit_tokens)

    def _apply(self, scores, allowed: set[int]):
        keep = np.fromiter(allowed, dtype=np.int64, count=len(allowed))

        # M3: read the pressure before destroying it.
        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        total = probs.sum()
        kept_mass = probs[keep].sum() if total > 0 else 0.0
        rejected = float(1.0 - kept_mass / total) if total > 0 else 0.0
        argmax_allowed = bool(int(np.argmax(scores)) in allowed)

        masked = np.full_like(scores, NEG_INF)
        masked[keep] = scores[keep]

        state = self.surface.trie.next_tokens(self._path)
        self.telemetry.steps.append(Step(
            allowed=len(allowed),
            rejected_mass=rejected,
            was_argmax=argmax_allowed,
            in_slot=state.spec.name if isinstance(state, SlotState) else None,
        ))
        return masked
