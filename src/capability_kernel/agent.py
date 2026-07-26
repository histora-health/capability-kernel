"""The production path: one model call, then the firmware decides.

The loop is deliberately small, and what it does *not* do is the design.

It does not plan. The domain computes which options exist, including the
combinations of chained arguments, and the model picks one — the law from
`token-trie` §4.2, which is also what keeps a coded procedure at one round trip
instead of three. At the measured ~2s per turn, three is six seconds for
something a dentist does several times per consultation.

It does not retry. A rejected call returns to the model as an error to correct,
and correcting badly after a rejection is the specific thing small models do:
14 malformed outputs in 30 turns, 0 of 10 tasks completed. So the surface is
narrowed until there is nothing invalid left to propose, and what survives goes
to the rules once.

And it does not execute. `propose` returns something a person confirms. A caller
that wants autonomy has to ask by calling `commit`, which is the right way
round for a system whose measured failure is a legal action on the wrong record.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .domain import Domain
from .firmware import Action, Context, Proposal, Runtime

SYSTEM = """\
You are a clinical records assistant.

Call exactly one tool, choosing from the options offered. The options are
computed from the current state of the record — they are the only ones that
exist right now, and anything not listed is not available.

If the request cannot be satisfied by any option offered, call `decline` and say
why. Do not substitute a different record for the one you were asked about.
"""


@dataclass
class Turn:
    """One request, and what came of it."""

    request: str
    proposal: Proposal | None = None
    #: What the model said when it called nothing. A refusal in prose is a real
    #: outcome, and distinguishing it from a failure to parse matters for the
    #: coverage gate.
    text: str = ""
    latency_s: float = 0.0
    #: Set when the model emitted something that was not a usable call. Counted
    #: separately from declining, because one is the model working and the other
    #: is the model failing.
    unparsed: str | None = None

    @property
    def proposed(self) -> bool:
        return self.proposal is not None


@dataclass
class Agent:
    """Domain, backend and firmware, wired into one turn.

    :param temperature: 0.0 by default. There is nothing to sample over when
        the options are enumerated — variety here is only a chance to pick a
        different one of the same finite set, which is not a feature.
    """

    domain: Domain
    backend: object
    runtime: Runtime
    temperature: float = 0.0
    max_tokens: int = 256
    history: list[Turn] = field(default_factory=list)

    def propose(self, request: str) -> Turn:
        """One request, one model call, one verdict. Nothing is executed."""
        started = time.time()
        turn = Turn(request=request)

        tools = self.domain.tool_schemas(self.runtime.store)
        reply = self.backend.chat(
            [{"role": "system", "content": SYSTEM + "\n\n" + self._state()},
             {"role": "user", "content": request}],
            tools, temperature=self.temperature, max_tokens=self.max_tokens)

        action = self._to_action(reply)
        if action is None:
            turn.text = reply["content"].strip()
            # A model that called nothing either refused or failed. The text is
            # the only evidence, so an empty one is the failure.
            turn.unparsed = None if turn.text else "no tool call and no text"
        else:
            turn.proposal = self.runtime.propose(
                action, Context(request=request, turn=len(self.history)))

        turn.latency_s = time.time() - started
        self.history.append(turn)
        return turn

    def commit(self, turn: Turn) -> str:
        """Execute a proposal a person approved."""
        if turn.proposal is None:
            raise ValueError("nothing was proposed for this turn")
        return self.runtime.commit(turn.proposal)

    # ── internals ────────────────────────────────────────────────────────────

    def _state(self) -> str:
        describe = getattr(self.runtime.store, "describe", None)
        return f"Current record:\n{describe()}" if describe else ""

    def _to_action(self, reply: dict) -> Action | None:
        """Turn a tool call into an action, expanding a chained choice.

        A chained method reaches the model as one enumerated `choice`, so the
        label it returns has to be expanded back into arguments. That expansion
        is recovery rather than validation — the value came from a list the
        program wrote — and a label that is not in the list means the model
        invented one, which is a failure rather than an argument to parse.
        """
        calls = reply.get("tool_calls") or []
        if not calls:
            return None

        fn = calls[0].get("function") or {}
        name = fn.get("name")
        if name not in self.domain.methods:
            return None

        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None

        if self.domain.is_chained(name) and "choice" in args:
            expanded = self.domain.parse_choice(name, args["choice"])
            if expanded not in self.domain.combinations(self.runtime.store, name):
                return None
            args = expanded

        return Action(method=name, args=args)
