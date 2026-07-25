"""Forcing the order of operations — the part a JSON Schema cannot do.

A schema says what shape an argument has. It cannot say *"this may only happen
after that"*, because that is not a statement about shape. Enumerating a
capability surface from live state can: the set of methods with a path through
the trie is recomputed at every step, so a method can stop existing and start
existing again as the world moves.

The rule demonstrated here is one every clinical system has and none can express
in a schema: **a change that has not been recorded admits nothing but recording
it.** Not "the harness rejects a second write" — while the audit is outstanding,
no token that begins a second write has a path.

    python examples/02_forced_order.py

No model. This shows what the surface permits, which is a property of the
compiled artefact — see `03_with_a_model.py` for whether a model then complies.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from capability_kernel import demo_store
from capability_kernel.compiler import ARM, compile_surface
from capability_kernel.manifest import enabled_methods, tool_schemas

TOKENIZER = "hf-internal-testing/tiny-random-gpt2"


def show(store, tok, tokenize, caption: str) -> None:
    """What the sampler may emit, and what a tool-calling harness would be told.

    Both are printed because they should agree. The mask and the schema compile
    from the same declaration, so a divergence between these two lines is a bug
    rather than a design choice.
    """
    surface = compile_surface(store, tokenize)
    allowed = enabled_methods(store)
    nxt = surface.trie.next_tokens([])

    # The trie holds every opcode; the phase controller decides which of them
    # the processor is allowed to reach right now.
    reachable = surface.indices_for(*allowed)
    words = sorted({surface.trie.opcodes[i][1] for i in reachable})

    print(f"\n   {caption}")
    print(f"     phase controller says : {allowed}")
    print(f"     reachable opcodes     : {len(reachable)} of {len(surface.trie)}")
    print(f"     methods with a path   : {words}")
    print(f"     tools a harness sees  : "
          f"{[t['function']['name'] for t in tool_schemas(store)]}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenize = lambda s: tok.encode(s, add_special_tokens=False)
    store = demo_store()

    print("A clinical folder, and one rule: nothing may be changed twice")
    print("without saying why in between.\n")
    print("=" * 70)

    show(store, tok, tokenize, "Nothing outstanding — the full surface")

    print("\n" + "=" * 70)
    print("\n   The model moves a file:")
    print(f"     {store.move('f_pano', 'std_endo')}")
    print(f"     store.pending_audit = {store.pending_audit!r}")

    show(store, tok, tokenize, "A change is unrecorded — the surface collapses")

    print("\n   Every other method just stopped existing. A second move is not")
    print("   a call that gets refused; there is no path that spells one.")
    print("   Note that `decline` is gone too — the model cannot refuse to")
    print("   record what it already did. That is the only place in this")
    print("   manifest where refusing is unreachable, and it has to be: a state")
    print("   you can decline your way out of is a state that permits exactly")
    print("   the thing the rule exists to prevent.")

    print("\n" + "=" * 70)
    print("\n   The model records it:")
    print(f"     {store.audit('was filed under the wrong study')}")

    show(store, tok, tokenize, "Recorded — the surface reopens")

    print("\n" + "=" * 70)
    print("\n   The journal, which is now guaranteed rather than hoped for:\n")
    for entry in store.journal:
        print(f"     {entry}")

    print("\n   That guarantee is the point. A validator can enforce the same")
    print("   rule by rejecting the second write — but a rejection is a call")
    print("   the model produced, that the harness caught, fed back, and hoped")
    print("   was retried correctly. Measured on gemma4:12b, the unmasked arm")
    print("   emitted 14 malformed calls and completed 0 of 10 tasks; retrying")
    print("   is not free.")

    print("\n   What this does NOT show: that a model complies. It shows what")
    print("   the surface permits. Compliance is 03_with_a_model.py.")


if __name__ == "__main__":
    main()
