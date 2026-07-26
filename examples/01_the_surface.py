"""Layer one: an option surface derived from live state.

Tools and argument enums are not fixed schemas. They are regenerated from the
state of the system, so a signed record disappears from the possible values the
moment it is signed — and what disappears has no path through the sampler, not a
rejection waiting downstream.

This runs in about a second and needs only a tokenizer, because what it shows is
a property of the compiled artefact rather than a statistic over samples. It
shows nothing about whether a model complies; that is 03.

    pip install -e ".[hf]"
    python examples/01_the_surface.py
"""

from __future__ import annotations

from transformers import AutoTokenizer

from capability_kernel import demo_store
from capability_kernel.compiler import ARM, body_text, compile_surface
from capability_kernel.manifest import surface_size

# Any tokenizer works. A tiny one keeps this example fast; swap in
# "google/gemma-4-E4B" to see the real numbers for the model you would deploy.
TOKENIZER = "hf-internal-testing/tiny-random-gpt2"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenize = lambda s: tok.encode(s, add_special_tokens=False)

    store = demo_store()
    print("The patient folder:\n")
    print(store.describe())

    # ── 1. The surface is a projection of the world ──────────────────────────
    print(f"\n{'─' * 70}\n1. What can be done right now\n")
    print(f"   {surface_size(store)}")
    print(f"   {sum(surface_size(store).values())} distinct actions, "
          f"enumerated from the store — not from a prompt.")

    surface = compile_surface(store, tokenize)

    # ── 2. At the choice point, the vocabulary is three tokens ───────────────
    print(f"\n{'─' * 70}\n2. What the sampler is allowed to emit\n")
    walk: list[int] = []
    for step in range(4):
        nxt = surface.trie.next_tokens(walk)
        if not isinstance(nxt, set) or not nxt:
            break
        words = sorted(tok.decode([t]).strip() for t in nxt)
        print(f"   step {step}: {len(nxt):>2} legal of {len(tok):,} — {words}")
        if len(nxt) > 1:
            break
        walk.append(next(iter(nxt)))

    print(f"\n   On gemma-4-E4B the same position offers 3 tokens of 262,144.")
    print(f"   That is the mechanism. Not a low probability — a short list.")

    # ── 3. A capability that does not exist has no path ──────────────────────
    print(f"\n{'─' * 70}\n3. What cannot be spelled\n")
    for attempt in (" delete", " rename target=std_hyg"):
        reachable = surface.trie.next_tokens(tokenize(attempt)) is not None
        verb = "reachable" if reachable else "NO PATH"
        print(f"   {ARM}{attempt!r:<28} {verb}")

    print("\n   `delete` is not in the manifest, so no token sequence spells it.")
    print("   `std_hyg` is a signed study, so no action can name it.")
    print("   Neither is rejected after the fact. Neither can be produced.")

    # ── 4. Naming is not acting ──────────────────────────────────────────────
    print(f"\n{'─' * 70}\n4. The exception, and why it exists\n")
    declines = [t for t, label in surface.trie.opcodes
                if label == "decline" and "std_hyg" in t]
    print(f"   {declines[0].strip()}")
    print("\n   A closed record IS nameable — exactly once, to be declined.")
    print("   Making it unnameable was measured causing the model to rename a")
    print("   different study instead: it had no way to say what it was asked")
    print("   about, so it said something it could say.")

    # ── 5. The surface follows the world ─────────────────────────────────────
    print(f"\n{'─' * 70}\n5. What happens when the world moves\n")
    before = surface_size(store)
    store.get("std_endo").signed = True
    after = surface_size(store)
    print(f"   before signing std_endo: {before}")
    print(f"   after:                   {after}")

    reduced = compile_surface(store, tokenize)
    can_move = any("std_endo" in t for t, label in reduced.trie.opcodes
                   if label == "move")
    print(f"\n   moving anything into std_endo still reachable? {can_move}")
    print("   The mask was recompiled from state. No prompt was updated, no")
    print("   validator was told. The capability stopped existing.")


if __name__ == "__main__":
    main()
