"""Compatibility shim over `resolvers` and `firmware.operand`.

The logic moved: resolution to `resolvers.LexicalResolver`, the decision to
`firmware.operand.check`, and the rule to `firmware.operand.operand_rule`. The
split exists because the rule is settled and the resolver is not.

Kept because the benchmarks and the mask experiment import these names, and
because a shim is cheaper to read than a diff across five files.
"""

from __future__ import annotations

from .firmware.operand import Mismatch, check as _check
from .resolvers import DEFAULT, Entity, Reference, Resolver

__all__ = ["Mismatch", "Reference", "Resolver", "check", "candidates", "resolve"]


def _entities(store):
    return [Entity(e.id, e.name) for e in store.nameable()]


def candidates(store, message: str) -> list[Reference]:
    return DEFAULT.candidates(_entities(store), message)


def resolve(store, message: str) -> Reference | None:
    """The single best referent, or None. See `candidates` for why a guard
    should rarely use this — a move names two records."""
    found = candidates(store, message)
    return found[0] if found else None


def check(store, message: str, target: str) -> Mismatch | None:
    return _check(_entities(store), message, target)
