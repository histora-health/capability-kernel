"""System-level control between what an agent proposes and what executes.

EAT called this layer Firmware and made it a string in a system prompt. This is
the same decomposition with a substrate: rules in AgentSpec's shape, evaluated
at the decision point, with nothing reaching the store except through `commit`.
"""

from .rules import Action, Context, Enforce, Rule, Trigger, Verdict, rule
from .runtime import Journal, Proposal, Refused, Runtime

__all__ = [
    "Action", "Context", "Enforce", "Rule", "Trigger", "Verdict", "rule",
    "Journal", "Proposal", "Refused", "Runtime",
]
