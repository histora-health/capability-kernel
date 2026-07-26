"""Block one: an option surface computed from live state.

Tools and argument enums are not fixed schemas. They are recomputed from the
state of the record each turn, so a signed study disappears from the possible
values the moment it is signed — not rejected downstream, absent from what the
model is offered.

No model, and it runs in well under a second, because what it shows is a
property of a computed artefact rather than a statistic over samples. It shows
nothing about whether a model complies; that is `03_the_agent.py`.

    PYTHONPATH=src python examples/01_the_surface.py
"""

from __future__ import annotations

from capability_kernel import CLINICAL, demo_store


def main() -> None:
    store = demo_store()
    print("The patient folder:\n")
    print(store.describe())

    # ── 1. The surface is a projection of the world ──────────────────────────
    print(f"\n{'─' * 70}\n1. What can be done right now\n")
    sizes = CLINICAL.surface_size(store)
    print(f"   {sizes}")
    print(f"   {sum(sizes.values())} distinct actions, enumerated from the "
          f"store — not from a prompt.")

    # ── 2. What the model is offered ─────────────────────────────────────────
    print(f"\n{'─' * 70}\n2. What the model is offered for `move`\n")
    for arg in ("target", "into"):
        values = CLINICAL.legal_values(store, "move", arg)
        print(f"   {arg:<8} {values}")
    print("\n   These are enums in the tool schema, not a described convention.")
    print("   The schema does not say `a string`; it says which records exist")
    print("   and are movable at this instant.")

    # ── 3. A capability that does not exist cannot be named ──────────────────
    print(f"\n{'─' * 70}\n3. What cannot be requested\n")
    print(f"   methods in the manifest : {tuple(CLINICAL.methods)}")
    print(f"   `delete` among them     : {'delete' in CLINICAL.methods}")
    signed = "std_hyg"
    for method in ("rename", "move", "set_metadata"):
        values = CLINICAL.legal_values(store, method, "target") or []
        print(f"   {signed} offered to {method:<13}: {signed in values}")

    print("\n   `delete` is not in the manifest, so no argument spells it.")
    print("   `std_hyg` is signed, so no acting method offers it as a target.")
    print("   Neither is rejected after the fact. Neither can be proposed.")

    # ── 4. Naming is not acting ──────────────────────────────────────────────
    print(f"\n{'─' * 70}\n4. The exception, and why it exists\n")
    print(f"   decline offers  : {CLINICAL.legal_values(store, 'decline', 'target')}")
    print("\n   A closed record IS nameable — exactly once, to be declined.")
    print("   Making it unnameable was measured causing the model to rename a")
    print("   different study instead: it had no way to say what it was asked")
    print("   about, so it said something it could say. That is the failure")
    print("   this whole architecture is built around.")

    # ── 5. The surface follows the world ─────────────────────────────────────
    print(f"\n{'─' * 70}\n5. What happens when the world moves\n")
    before = CLINICAL.surface_size(store)
    store.get("std_endo").signed = True
    after = CLINICAL.surface_size(store)
    print(f"   before signing std_endo : {before}")
    print(f"   after                   : {after}")

    reachable = CLINICAL.legal_values(store, "move", "into") or []
    print(f"\n   std_endo still a destination? {'std_endo' in reachable}")
    print("   The surface was recomputed from state. No prompt was updated and")
    print("   no validator was told. The capability stopped existing.")


if __name__ == "__main__":
    main()
