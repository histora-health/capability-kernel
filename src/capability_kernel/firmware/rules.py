"""Rules, in the shape AgentSpec established.

`trigger`, `check`, `enforce` — a DSL evaluated at ICSE 2026 across code agents,
embodied agents and autonomous driving, with millisecond overhead. We take the
shape and not the surface syntax: a language exists so non-engineers can author
rules, and for a proof of concept that is premature. These are Python objects
with callable predicates carrying the same three fields, so a surface syntax
remains a later question rather than a rewrite.

Two departures from the paper, each because something was measured.

The check receives the **request** alongside the action and the store. AgentSpec
predicates reason about the action; the failure this repository measured needs a
predicate that reasons about the relation between the action and what was asked
for, and that relation is not visible from the action alone.

And `inspect` is the default for anything uncertain rather than a special case.
The measured failure is silent — a legal action on a record nobody named, with
no violation to log — so a rule that cannot decide must surface the proposal
rather than let it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol


class Trigger(str, Enum):
    """When a rule is evaluated.

    Only `BEFORE_ACTION` is used today. The other two are declared because
    AgentSpec has them and the omission would be a silent narrowing of the
    model rather than a decision — a rule that fires when the world changes, or
    when a task ends, is a different thing from one that gates a call.
    """

    BEFORE_ACTION = "before_action"
    ON_STATE_CHANGE = "on_state_change"
    ON_TASK_COMPLETE = "on_task_complete"


class Enforce(str, Enum):
    """What happens when a rule fires.

    `INSPECT` is not a weaker `BLOCK`. Blocking says the action is wrong;
    inspecting says the system cannot tell, which is a different claim and the
    honest one for operand mismatches — the request may simply have been
    phrased in a way the resolver does not cover.
    """

    BLOCK = "block"
    INSPECT = "inspect"
    SUBSTITUTE = "substitute"


@dataclass(frozen=True)
class Action:
    """What the model proposed. Not yet executed."""

    method: str
    args: dict

    @property
    def target(self) -> str | None:
        """The entity acted upon, by the manifest's convention.

        Returned rather than assumed present: `decline` names a target and a
        method with no operand does not, and a rule that reads `args["target"]`
        directly breaks on the second kind.
        """
        return self.args.get("target")

    def __str__(self) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.method} {body}".strip()


@dataclass(frozen=True)
class Context:
    """What the rule needs beyond the action itself.

    `request` is the user's words, and it is here because operand verification
    cannot work without them. Carrying it explicitly rather than reaching into
    conversation history keeps a rule a pure function of its inputs, which is
    what makes it testable without a model.
    """

    request: str
    turn: int = 0


class Predicate(Protocol):
    """Returns something truthy when the rule should fire.

    Truthy rather than boolean on purpose: a predicate may return an object
    explaining *why* — `reference.check` returns a `Mismatch` — and that
    explanation is what a person reads when the enforcement is `INSPECT`.
    """

    def __call__(self, action: Action, store, context: Context): ...


@dataclass(frozen=True)
class Rule:
    """One constraint, in AgentSpec's three parts."""

    id: str
    check: Predicate
    enforce: Enforce = Enforce.BLOCK
    trigger: Trigger = Trigger.BEFORE_ACTION
    #: Only consulted for SUBSTITUTE. A rule that replaces an action without
    #: saying what with is a rule that silently drops it.
    substitute: Action | None = None
    #: Lower runs first. Authority before ordering before operand, so a person
    #: is never asked to inspect something that was going to be blocked anyway.
    priority: int = 100
    description: str = ""

    def __post_init__(self) -> None:
        if self.enforce is Enforce.SUBSTITUTE and self.substitute is None:
            raise ValueError(
                f"rule {self.id!r} substitutes without saying what with")

    def evaluate(self, action: Action, store, context: Context):
        """Whatever the predicate returned, or None if it did not fire."""
        return self.check(action, store, context) or None


@dataclass
class Verdict:
    """The outcome of evaluating every rule against one proposed action."""

    action: Action
    enforce: Enforce | None = None
    #: The rules that fired, in the order they fired. Plural because a person
    #: reading an inspection needs to know everything that objected, not the
    #: first thing.
    fired: list[tuple[Rule, object]] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.enforce is None

    @property
    def blocked(self) -> bool:
        return self.enforce is Enforce.BLOCK

    @property
    def needs_inspection(self) -> bool:
        return self.enforce is Enforce.INSPECT

    @property
    def reasons(self) -> list[str]:
        return [str(detail) if detail is not True else (rule.description or rule.id)
                for rule, detail in self.fired]

    def __str__(self) -> str:
        if self.allowed:
            return f"allow {self.action}"
        return f"{self.enforce.value} {self.action} — " + "; ".join(self.reasons)


def rule(id: str, check: Callable, **kwargs) -> Rule:
    """Terser constructor, for manifests that declare many rules."""
    return Rule(id=id, check=check, **kwargs)
