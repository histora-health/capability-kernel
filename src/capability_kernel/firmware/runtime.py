"""The decision point: where a proposed action meets the rules.

Everything the agent wants to do passes through `propose`, and nothing reaches
the store except through `commit`. That is the whole of the layer — the value is
not in what it computes but in there being exactly one place where an action can
become an effect.

Enforcement happens here rather than at the sampler because that is where the
measurements pointed: post-hoc validation was never defeated in any arm of the
first phase, while sampler enforcement introduced a failure of its own. The
mask remains in `experiments/` for anyone who wants to check that claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import Action, Context, Enforce, Rule, Trigger, Verdict


@dataclass
class Proposal:
    """An action that passed the rules but has not been executed.

    It exists because `INSPECT` needs something to hand a person, and because
    the thing they are shown must be the thing that later runs — a proposal
    re-derived at commit time could differ from the one that was approved.
    """

    action: Action
    verdict: Verdict
    context: Context

    @property
    def target_name(self) -> str | None:
        """Set by the runtime. Interfaces show this, never the identifier.

        A person confirming `f_pa11` is confirming a string. A person
        confirming *periapical_11.dcm* is confirming a record, and the failure
        this whole layer exists for is the one where those differ.
        """
        return self._target_name

    _target_name: str | None = None


@dataclass
class Journal:
    """What was proposed, what fired, and what happened.

    Separate from the store's own journal, which records effects. This records
    *decisions*, including the ones that produced no effect — a blocked action
    leaves no trace in the data, and it is exactly the trace an audit wants.
    """

    entries: list[dict] = field(default_factory=list)

    def record(self, kind: str, verdict: Verdict, context: Context, **extra) -> None:
        self.entries.append({
            "kind": kind,
            "turn": context.turn,
            "request": context.request,
            "action": str(verdict.action),
            "enforce": verdict.enforce.value if verdict.enforce else None,
            "rules": [r.id for r, _ in verdict.fired],
            "reasons": verdict.reasons,
            **extra,
        })

    def __len__(self) -> int:
        return len(self.entries)

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e["kind"] == kind]


class Runtime:
    """Evaluates rules against proposed actions, and executes what survives.

    :param store: the domain. Rules read it; `commit` writes through it.
    :param rules: evaluated in priority order. Authority before ordering before
        operand, so nobody is asked to inspect something that was going to be
        blocked anyway.
    :param execute: how an allowed action reaches the world. Defaults to calling
        the method on the store, which is what the clinical domain does;
        injectable so a domain whose actions are not store methods can use this
        without pretending they are.
    """

    def __init__(self, store, rules: list[Rule], *, execute=None) -> None:
        self.store = store
        self.rules = sorted(rules, key=lambda r: r.priority)
        self.journal = Journal()
        self._execute = execute or self._call_store

    # ── The decision point ───────────────────────────────────────────────────

    def evaluate(self, action: Action, context: Context,
                 trigger: Trigger = Trigger.BEFORE_ACTION) -> Verdict:
        """Run every rule for this trigger. Does not stop at the first hit.

        A person inspecting a proposal needs to know everything that objected,
        not the first thing — so all matching rules are collected and the
        strictest enforcement wins.
        """
        verdict = Verdict(action=action)

        for rule in self.rules:
            if rule.trigger is not trigger:
                continue
            detail = rule.evaluate(action, self.store, context)
            if detail is None:
                continue
            verdict.fired.append((rule, detail))
            verdict.enforce = _strictest(verdict.enforce, rule.enforce)

        return verdict

    def propose(self, action: Action, context: Context) -> Proposal:
        """Evaluate, and return something the interface can act on.

        Nothing is executed here, including when every rule passes. Separating
        proposal from commit is what makes propose-and-confirm the default
        rather than a mode — a caller that wants autonomy has to ask for it by
        calling `commit`, which is the right way round.
        """
        verdict = self.evaluate(action, context)
        entity = self.store.get(action.target) if action.target else None
        proposal = Proposal(action=action, verdict=verdict, context=context,
                            _target_name=getattr(entity, "name", None))
        self.journal.record("proposed", verdict, context,
                            target_name=proposal.target_name)
        return proposal

    def commit(self, proposal: Proposal) -> str:
        """Execute a proposal a person approved.

        The rules are re-evaluated. Between proposal and confirmation the world
        may have moved — another user may have signed the study being renamed —
        and approving a proposal is approving *that* action, not a licence to
        run it against a state nobody saw.
        """
        fresh = self.evaluate(proposal.action, proposal.context)
        if not fresh.allowed:
            self.journal.record("stale", fresh, proposal.context)
            raise Refused(f"state changed since the proposal: {fresh}")

        result = self._execute(proposal.action)
        self.journal.record("committed", fresh, proposal.context, result=result)
        return result

    def run(self, action: Action, context: Context) -> str:
        """Propose and commit in one step, for callers that are not asking a
        person. Blocks and inspections both raise, because an autonomous caller
        has nobody to show an inspection to."""
        proposal = self.propose(action, context)
        if not proposal.verdict.allowed:
            raise Refused(str(proposal.verdict))
        return self.commit(proposal)

    # ── Execution ────────────────────────────────────────────────────────────

    def _call_store(self, action: Action) -> str:
        method = getattr(self.store, action.method, None)
        if method is None:
            # Unreachable when the option surface is derived from the manifest,
            # and loud rather than silent if the two ever diverge.
            raise Refused(f"{action.method!r} is not an operation of this store")
        return method(**action.args)


class Refused(Exception):
    """An action that the rules did not allow, or that went stale."""


def _strictest(current: Enforce | None, incoming: Enforce) -> Enforce:
    """BLOCK beats INSPECT beats SUBSTITUTE beats nothing.

    Ordered by how much the system is claiming. Blocking claims the action is
    wrong; inspecting claims it cannot tell. When one rule is certain and
    another is not, the certain one decides.
    """
    order = {Enforce.SUBSTITUTE: 1, Enforce.INSPECT: 2, Enforce.BLOCK: 3}
    if current is None:
        return incoming
    return current if order[current] >= order[incoming] else incoming
