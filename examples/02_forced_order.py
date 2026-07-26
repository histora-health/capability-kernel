"""Block two: order forced at the system level.

This is the class of rule a JSON Schema structurally cannot state. A schema says
what shape an argument has; it cannot say *"this may only happen after that"*,
because that is not a statement about shape. A surface computed from live state
can, because the set of available methods is recomputed every turn.

The rule shown here is one every clinical system has: **a change that has not
been recorded admits nothing but recording it.** Not "the harness rejects the
second write" — while the audit is outstanding, a second write is not among the
tools the model is given.

    PYTHONPATH=src python examples/02_forced_order.py

No model. This shows what the surface permits, which is a property of a computed
artefact. Whether a model then complies is `03_the_agent.py`.
"""

from __future__ import annotations

from capability_kernel import CLINICAL, demo_store


def show(store, caption: str) -> None:
    print(f"\n   {caption}")
    print(f"     methods available : {CLINICAL.enabled_methods(store)}")
    print(f"     tools the model sees : "
          f"{[t['function']['name'] for t in CLINICAL.tool_schemas(store)]}")
    print(f"     distinct actions  : {sum(CLINICAL.surface_size(store).values())}")


def main() -> None:
    store = demo_store()

    print("A clinical folder, and one rule: nothing may be changed twice")
    print("without saying why in between.\n")
    print("=" * 70)

    show(store, "Nothing outstanding — the full surface")

    print("\n" + "=" * 70)
    print("\n   The model moves a file:")
    print(f"     {store.move('f_pano', 'std_endo')}")
    print(f"     store.pending_audit = {store.pending_audit!r}")

    show(store, "A change is unrecorded — the surface collapses")

    print("\n   Every other method just stopped existing. A second move is not")
    print("   a call that gets refused; it is not among the tools offered.")
    print("   Note that `decline` is gone too — the model cannot refuse to")
    print("   record what it already did. That is the only place in this")
    print("   manifest where refusing is unavailable, and it has to be: a state")
    print("   you can decline your way out of is a state that permits exactly")
    print("   the thing the rule exists to prevent.")

    print("\n" + "=" * 70)
    print("\n   The model records it:")
    print(f"     {store.audit('was filed under the wrong study')}")

    show(store, "Recorded — the surface reopens")

    print("\n" + "=" * 70)
    print("\n   The journal, which is now guaranteed rather than hoped for:\n")
    for entry in store.journal:
        print(f"     {entry}")

    print("\n   That guarantee is the point. A validator enforces the same rule")
    print("   by rejecting the second write — but a rejection is a call the")
    print("   model produced, that the harness caught, fed back, and hoped was")
    print("   retried correctly. Correcting badly after a rejection is the")
    print("   specific thing small models do, which is why the surface narrows")
    print("   instead: 14 malformed calls in 30 turns, 0 of 10 tasks completed.")

    print("\n   The same ordering is also enforced as a firmware rule at the")
    print("   decision point — see `firmware/clinical.py`, `audit_first`.")
    print("   Belt and braces on purpose: the surface is the usability")
    print("   mechanism, the rule is the one that has to hold.")


if __name__ == "__main__":
    main()
