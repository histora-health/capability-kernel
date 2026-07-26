"""A firmware layer between what an agent proposes and what the world executes.

Four blocks, none of which depends on the model being clever enough to avoid
the error:

* :mod:`domain` — the option surface, computed from live state each turn, so a
  signed record leaves the choices the moment it is signed
* :mod:`firmware` — rules at the decision point, including order, which is the
  one class of constraint a JSON Schema structurally cannot state
* :mod:`firmware.operand` — verification that the action names the record the
  *request* named, which is the failure the rest of this does not catch
* :class:`agent.Agent` — one model call, then a proposal a person confirms

The package used to open by claiming that an unauthorised action is unemittable
because the sampler never offers the tokens. That thesis is retired: post-hoc
validation was never defeated, and masking the sampler produced silent
substitution instead. The code and the measurements that retired it are kept
runnable in ``experiments/mask/``, because the argument for this architecture is
that the mask did not carry it.
"""

from .agent import Agent, Turn
from .domain import Domain, Method
from .firmware import Action, Context, Journal, Proposal, Refused, Runtime, Verdict
from .manifest import CLINICAL, tool_schemas
from .store import ClinicalStore, StoreError, demo_store

__all__ = [
    "Agent", "Turn",
    "Domain", "Method",
    "Action", "Context", "Journal", "Proposal", "Refused", "Runtime", "Verdict",
    "ClinicalStore", "StoreError", "demo_store",
    "CLINICAL", "tool_schemas",
]
