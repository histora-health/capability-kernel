"""Capabilities applied at the sampler.

An agent cannot emit an action it is not authorised to take — not rejected,
not detected, but absent from the sampler's candidate set at the step where
the model would have chosen it.

Two arms share one manifest:

* :mod:`harness` — the model is *asked* to stay inside the surface (baseline)
* the token trie — the model *cannot* leave it (enforced)
"""

from .manifest import MANIFEST, opcode_strings, surface_size, tool_schemas
from .chat import EnforcedChat, Violation
from .store import ClinicalStore, StoreError, demo_store

__all__ = [
    "EnforcedChat", "Violation",
    "ClinicalStore", "StoreError", "demo_store",
    "MANIFEST", "tool_schemas", "opcode_strings", "surface_size",
]
