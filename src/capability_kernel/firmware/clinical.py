"""The rules the clinical domain needs, in the three families.

Authority, ordering and operand — each with the priority that keeps a person
from being asked to inspect something that was going to be blocked anyway.

They are written here rather than generated from the manifest because a rule
and a surface answer different questions. The surface decides what the model
may *propose*; a rule decides what may *happen*. Most of the time they agree,
and the interesting cases are where they do not: a request naming a record the
surface never offered is exactly the failure this layer was built for.
"""

from __future__ import annotations

from ..manifest import CLINICAL
from ..resolvers import Entity
from .operand import operand_rule
from .rules import Action, Context, Enforce, Rule, Trigger


def _closed(action: Action, store, _: Context):
    """A signed study, or anything inside one, may not be changed."""
    if action.method in ("decline", "audit") or not action.target:
        return None
    entity = store.get(action.target)
    if entity is None:
        return f"{action.target!r} does not exist"
    folder = getattr(entity, "folder_id", None)
    if getattr(entity, "signed", False) or (folder and store.get(folder).signed):
        return f"{action.target!r} is in a signed study and cannot be changed"
    return None


def _audit_first(action: Action, store, _: Context):
    """An unrecorded change admits nothing but recording it."""
    if store.pending_audit and action.method != "audit":
        return (f"{store.pending_audit!r} has not been recorded; "
                f"an audit entry must come first")
    return None


def clinical_entities(store) -> list[Entity]:
    """Every record a request could refer to, closed ones included.

    Closed records must resolve, or the rule is blind exactly where the failure
    happens: substitution follows a request about a record nothing may be done
    to, so a resolver that could not see them would return nothing precisely
    when it matters.
    """
    return [Entity(e.id, e.name) for e in store.nameable()]


CLINICAL_RULES = [
    Rule("closed_record", _closed, Enforce.BLOCK, Trigger.BEFORE_ACTION,
         priority=10,
         description="a signed study and its files cannot be changed"),
    Rule("audit_first", _audit_first, Enforce.BLOCK, Trigger.BEFORE_ACTION,
         priority=20,
         description="an unrecorded change must be audited before anything else"),
    operand_rule(clinical_entities, exempt=("decline", "audit")),
]
