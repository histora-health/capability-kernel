"""The rule that asks whether the action names what the request named.

AgentSpec's rules reason about the action: is this call permitted, in this
state, by this agent. This one reasons about the **relation between the action
and the request**, which is a different question and the one that catches silent
substitution — an action every other check passes because it is entirely legal,
performed on a record nobody mentioned.

It is enforced as `INSPECT` rather than `BLOCK`, and the distinction carries the
whole design. Blocking claims the action is wrong. This claims the system cannot
tell: the request may simply have been phrased in a way the resolver does not
cover, and a rule that blocks legitimate work is a rule that gets switched off,
after which it protects nothing.

The resolver is injected because its logic is the unsettled part. String overlap
catches every case measured so far and will not survive cross-language
reference; when it is replaced, this file does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..resolvers import DEFAULT, Entity, Reference, Resolver
from .rules import Action, Context, Enforce, Rule, Trigger


@dataclass
class Mismatch:
    """The action names a record the request does not.

    `named` is plural because requests usually are: a move mentions what to move
    and where to put it, and reporting only the strongest match would tell a
    reviewer the request was about the destination.
    """

    named: tuple[Entity, ...]
    acted_on: Entity
    matched: tuple[str, ...]

    @property
    def asked_about(self) -> Entity:
        return self.named[0]

    def __str__(self) -> str:
        names = ", ".join(e.id for e in self.named)
        return (f"the request refers to {names} — not to {self.acted_on.id!r}, "
                f"which this action would change. Words matched: "
                f"{list(self.matched)}")


def check(entities, message: str, target: str, *,
          resolver: Resolver = DEFAULT) -> Mismatch | None:
    """Whether acting on `target` contradicts what `message` referred to.

    Returns None — meaning no objection — in three cases, and each one is a
    decision rather than an omission.

    **Nothing was named.** A request with no referent cannot have a wrong one.
    Objecting here would fire on every open-ended instruction.

    **The target is among the named.** Including when it is not the strongest
    match, because a move names two records and only one of them is the target.

    **The target does not exist.** That is a different failure and a different
    rule's job; reporting it here would put two diagnoses behind one message.
    """
    entities = list(entities)
    named = resolver.candidates(entities, message)
    if not named:
        return None
    if any(r.entity.id == target for r in named):
        return None

    acted = next((e for e in entities if e.id == target), None)
    if acted is None:
        return None

    matched = tuple(sorted({w for r in named for w in r.matched}))
    return Mismatch(tuple(r.entity for r in named), acted, matched)


def operand_rule(nameable, *, resolver: Resolver = DEFAULT,
                 exempt: tuple[str, ...] = (), priority: int = 30) -> Rule:
    """Build the rule for a domain.

    :param nameable: ``store -> Iterable[Entity]``. The domain supplies it
        because only the domain knows how its records are named, and the
        resolver should not have to know either.
    :param exempt: methods with no operand to verify. `decline` names a record
        in order to refuse it, which is the opposite of substituting for it, and
        `audit` names the change rather than a record.
    """

    def check_action(action: Action, store, context: Context):
        if action.method in exempt or not action.target:
            return None
        return check(nameable(store), context.request, action.target,
                     resolver=resolver)

    return Rule(
        id="operand_matches",
        check=check_action,
        enforce=Enforce.INSPECT,
        trigger=Trigger.BEFORE_ACTION,
        priority=priority,
        description="the action names a record the request does not",
    )
