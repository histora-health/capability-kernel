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
    :param arm: the word that hands control to the mask. Matched against decoded
        text rather than a token sequence, because how it tokenizes depends on
        what precedes it and the model picks that.
    :param close_tokens: the tokens that terminate an action line, used as the
        clamp when the walk desynchronises.
    :param detokenize: needed for slots — whether a token is legal inside a free
        argument depends on what it *says*, which only the tokenizer knows.
    :param enabled: opcode indices the phase controller currently permits. None
        means all of them. This is what makes the surface a function of state
        rather than a fixed grammar: a phase that forbids moving does not
        validate moves and reject them, it compiles a trie without them.
    """

    def __init__(self, surface: CompiledSurface, arm: str,
                 detokenize, close_tokens: list[int], *,
                 enabled: set[int] | None = None,
                 telemetry: Telemetry | None = None) -> None:
        self.surface = surface
        self.arm = arm
        self.detokenize = detokenize
        #: What to emit when there is nothing safe left to emit. Required
        #: rather than derived: a mask with no legal token stalls the sampler
        #: at negative infinity across the whole vocabulary.
        self.close_tokens = list(close_tokens)
        self.enabled = enabled if enabled is not None else surface.all_indices
        self.telemetry = telemetry if telemetry is not None else Telemetry()

        # None means "not armed"; [] means "armed, at the root". The two are
        # different states and conflating them disarms the mask on the very
        # step that arms it.
        self._path: list[int] | None = None
        self._prompt_len: int | None = None
        #: Set when the walk leaves the trie. Not an exception: see _allowed_now.
        self.desynchronised: str | None = None

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._path is not None

    def reset(self) -> None:
        self._path, self._prompt_len = None, None

    def _generated(self, input_ids) -> list[int]:
        if self._prompt_len is None:
            self._prompt_len = len(input_ids)
        return list(input_ids[self._prompt_len:])

    def _locate(self, gen: list[int]) -> list[int] | None:
        """Where the current action starts, derived from what was generated.

        Deliberately not tracked incrementally. llama.cpp does not tell a
        processor which token the sampler chose, so a hand-maintained path
        needs the caller to feed every token back — two sources of truth that
        drift the moment anything retries, rewinds or batches. ``input_ids``
        already carries the answer.

        The arming word is found by decoding rather than by matching token ids.
        On gemma4 the same word tokenizes differently after a newline than
        after ``<channel|>``, and matching ids missed the second — which let a
        whole ``delete`` line through unmasked.
        """
        for i in range(len(gen), -1, -1):
            try:
                head = self.detokenize(gen[:i])
            except Exception:
                continue
            if not head.endswith(self.arm):
                continue
            path = gen[i:]
            # A finished action releases the mask; prose after it is free until
            # the model arms again.
            #
            # The question is whether the action ended, not whether the whole
            # remaining path is an opcode. Those differ the moment one token
            # arrives after a completed action: the path stops being complete,
            # so the mask treated finished work as still in progress and
            # reported desynchronisation on every successful action. Measured
            # on gemma4, which emits <|channel|> straight after the closing
            # newline.
            for k in range(1, len(path) + 1):
                if self.surface.trie.is_complete(path[:k]):
                    return None
            return path
        return None

    # ── The mask ─────────────────────────────────────────────────────────────

    def __call__(self, input_ids, scores):
        gen = self._generated(input_ids)
        was_active = self._path is not None
        self._path = self._locate(gen)

        if self._path is None:
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
            # The walk left the trie: the sampler and the trie disagree about
            # what was emitted.
            #
            # This used to raise. It must not. llama.cpp calls the processor
            # through a ctypes callback, which *swallows* the exception —
            # printing "Exception ignored" and then continuing to generate with
            # no mask at all. A design meant to fail loud failed open, which is
            # the worst outcome available and was only visible because the run
            # produced a wrong write afterwards.
            #
            # So instead: record it, and clamp generation to the tokens that end
            # the action. Nothing further can be emitted, and the caller checks
            # the flag and discards the turn.
            self.desynchronised = (
                f"the walk left the trie after {len(self._path)} tokens; "
                f"tokenizer parity has broken — re-run parity_report"
            )
            return self._closers()

        if isinstance(nxt, SlotState):
            return self._slot_tokens(nxt, scores)

        return {t for t in nxt if self._reaches_enabled(t)}

    def _closers(self) -> set[int]:
        """The only tokens left when the walk is lost.

        Whatever comes next cannot be more of an action. Falling back to the
        closing token ends the line; the caller then sees ``desynchronised``
        and discards the turn rather than trusting a half-masked action.
        """
        return set(self.close_tokens)

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
