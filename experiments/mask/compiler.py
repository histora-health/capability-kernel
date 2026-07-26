"""Manifest + store → token trie.

This is the compilation step the article describes: a declarative capability
surface becomes a mask applied inside the decoding loop. It is model-specific,
because it depends on the tokenizer — which is why :func:`parity_report` exists
and why it runs before anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

from capability_kernel.manifest import MANIFEST, legal_values
from capability_kernel.store import ClinicalStore
from trie import SlotSpec, TokenTrie

#: Wraps every action. Chosen for tokenizer-friendliness rather than
#: aesthetics: ordinary punctuation that every BPE vocabulary splits the same
#: way in isolation and in context. Exotic markers like ``<|call|>`` look
#: tidier and fragment differently per model, which is how a trie ends up
#: never intersecting top-K.
#:
#: Note the absent trailing space. On gemma4 ``"\nACTION "`` tokenizes to
#: ``['\n', 'ACTION', ' ']`` alone but ``['\n', 'ACTION', ' move', ...]`` in
#: context — BPE pulls the space into the following word. Any boundary the trie
#: cuts at must not sit before a variable segment, or the walk desynchronises
#: from the sampler. :func:`parity_report` tests exactly this.
OPEN = "\nACTION"
CLOSE = "\n"

#: What actually arms the mask, and deliberately not the same as :data:`OPEN`.
#:
#: A frame the model has to spell exactly is a trigger the model controls, and
#: it can decline to spell it. Measured on gemma4:12b: asked to delete a file,
#: it emitted ``<channel|>ACTION delete(path='std_hyg/f_chart')`` — no leading
#: newline, so a ``"\nACTION"`` trigger never fired and the whole line was
#: generated free. Arming on the bare word instead means every occurrence of
#: ACTION, wherever it appears, hands control to the trie. Fail-closed: the
#: cost is that the model cannot discuss actions in prose, which is a smaller
#: problem than being able to opt out of enforcement.
ARM = "ACTION"

#: Characters that may not appear inside a free-text argument, because they
#: would close it early or break the line-oriented frame.
SLOT_FORBID = "\n\r\"'()="


@dataclass
class CompiledSurface:
    trie: TokenTrie
    #: method -> opcode indices, for phase control.
    by_method: dict[str, set[int]]
    #: How many opcodes each method contributed. Watch this: enumeration has a
    #: ceiling and a busy folder will find it.
    sizes: dict[str, int]

    def indices_for(self, *methods: str) -> set[int]:
        out: set[int] = set()
        for m in methods:
            out |= self.by_method.get(m, set())
        return out

    @property
    def all_indices(self) -> set[int]:
        return self.indices_for(*self.by_method)


def action_text(method: str, args: dict[str, str]) -> str:
    """The wire form of one action.

    ``ACTION move target=f_pano into=std_endo``

    Deliberately not JSON. JSON's quoting and nesting give a slot several ways
    to end, and every one of them is a place the trie and the tokenizer can
    disagree.
    """
    return OPEN + body_text(method, args)


def body_text(method: str, args: dict[str, str]) -> str:
    """The action from the arming word onward — what the trie actually holds.

    ``ACTION`` itself is not in the trie. Whether BPE gives it one token or
    three depends on what precedes it, and the model chooses that: measured on
    gemma4 it appeared after a newline and after ``<channel|>``. Rooting the
    trie one token later makes the walk independent of that choice.
    """
    body = " ".join(f"{k}={v}" for k, v in args.items())
    return f" {method} {body}{CLOSE}"


def compile_surface(store: ClinicalStore, tokenize, *,
                    slot_max_tokens: int = 16,
                    max_per_method: int = 512) -> CompiledSurface:
    """Build the trie for the world as it is right now.

    :param tokenize: ``str -> list[int]`` using the *active model's* tokenizer.
    :param max_per_method: refuse rather than silently stall. Every opcode costs
        a tokenize call at build time, so a wide enumeration is a pause before
        the first token, not a gradual slowdown.
    """
    trie = TokenTrie()
    by_method: dict[str, set[int]] = {}
    sizes: dict[str, int] = {}

    for method, spec in MANIFEST.items():
        combos = _enumerate(store, method, spec)
        if len(combos) > max_per_method:
            raise ValueError(
                f"{method} would compile to {len(combos)} opcodes, over the ceiling "
                f"of {max_per_method}. Narrow the surface or move the wide argument "
                f"to a slot."
            )

        indices: set[int] = set()
        for args in combos:
            slot_arg = next((k for k, v in args.items() if v is None), None)

            if slot_arg is None:
                text = action_text(method, args)
                indices.add(trie.insert(text, tokenize(body_text(method, args)), method))
                continue

            # Split the action at the free argument: everything up to and
            # including "key=" is fixed, the value is a slot, the rest is fixed.
            before, after = _split_at(args, slot_arg)
            prefix = f" {method} " + "".join(f"{k}={v} " for k, v in before) + f"{slot_arg}="
            suffix = ("".join(f" {k}={v}" for k, v in after)) + CLOSE
            spec_slot = SlotSpec(forbid=SLOT_FORBID, max_tokens=slot_max_tokens,
                                 name=f"{method}.{slot_arg}")
            indices.add(trie.insert_with_slot(
                prefix + "{slot}" + suffix, tokenize(prefix), spec_slot,
                tokenize(suffix), method))

        by_method[method] = indices
        sizes[method] = len(combos)

    return CompiledSurface(trie=trie, by_method=by_method, sizes=sizes)


def _enumerate(store: ClinicalStore, method: str, spec) -> list[dict]:
    """Cartesian product of the enumerable arguments; free text stays ``None``."""
    combos: list[dict] = [{}]
    for arg in spec.args:
        values = legal_values(store, method, arg)
        nxt = []
        for combo in combos:
            if values is None:
                nxt.append({**combo, arg: None})
            else:
                for v in values:
                    nxt.append({**combo, arg: v})
        combos = nxt
    return combos


def _split_at(args: dict, key: str):
    items = list(args.items())
    i = [k for k, _ in items].index(key)
    return items[:i], items[i + 1:]


# ── M0: does this model tokenize the surface stably? ─────────────────────────


def parity_report(store: ClinicalStore, tokenize, detokenize) -> dict:
    """Check the surface survives a tokenize/detokenize round trip.

    The failure this catches is subtle and fatal: a string that tokenizes one
    way in isolation and another way in context puts the trie out of step with
    the sampler, and the symptom is not an error — it is a model that emits
    valid syntax and stops choosing.

    Run before anything else. If a model fails this, no amount of downstream
    work fixes it.
    """
    checked, failures = 0, []

    for method, spec in MANIFEST.items():
        for args in _enumerate(store, method, spec):
            concrete = {k: (v if v is not None else "example_name") for k, v in args.items()}
            text = action_text(method, concrete)
            tokens = tokenize(text)
            back = detokenize(tokens)
            checked += 1
            if back != text:
                failures.append({"method": method, "text": text, "round_trip": back})

    # The framing tokens matter most: they appear in every single action.
    frame = {}
    for label, piece in (("open", OPEN), ("close", CLOSE)):
        toks = tokenize(piece)
        frame[label] = {"text": piece, "tokens": len(toks),
                        "round_trip_ok": detokenize(toks) == piece}

    unstable = prefix_stability(store, tokenize)

    return {
        "checked": checked,
        "failures": failures,
        "unstable_prefixes": unstable,
        "ok": (not failures and not unstable
               and all(f["round_trip_ok"] for f in frame.values())),
        "frame": frame,
    }


def prefix_stability(store, tokenize) -> list[dict]:
    """Every boundary the trie cuts at must survive being cut at.

    This is the property the trie actually depends on, and round-trip fidelity
    does not imply it. ``"\nACTION "`` round-trips perfectly and still
    tokenizes differently in context, because BPE merges the trailing space
    into the following word. A trie built on the whole string but walked from a
    separately-tokenized prefix then desynchronises from the sampler — and the
    symptom is not an exception, it is a mask that never matches.

    Returns the boundaries that failed. Empty means the surface is safe to
    compile for this tokenizer.
    """
    bad: list[dict] = []

    def check(where: str, prefix: str, whole: str) -> None:
        pt, wt = tokenize(prefix), tokenize(whole)
        if wt[:len(pt)] != pt:
            bad.append({"where": where, "prefix": prefix,
                        "isolated": pt, "in_context": wt[:len(pt) + 1]})

    for method, spec in MANIFEST.items():
        for args in _enumerate(store, method, spec):
            concrete = {k: (v if v is not None else "example_name") for k, v in args.items()}
            whole = body_text(method, concrete)

            # Every boundary the compiler cuts at, now rooted at the body.
            check(f"{method}:method", f" {method}", whole)
            slot_arg = next((k for k, v in args.items() if v is None), None)
            if slot_arg is not None:
                before, _ = _split_at(args, slot_arg)
                prefix = (f" {method} "
                          + "".join(f"{k}={concrete[k]} " for k, _ in before)
                          + f"{slot_arg}=")
                check(f"{method}:slot:{slot_arg}", prefix, whole)
    return bad
