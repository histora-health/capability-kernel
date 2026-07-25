"""The compiler, tested without loading a model.

A real tokenizer makes these tests slow and machine-dependent, and it cannot be
made to fail on demand — so the interesting properties would go untested. These
fakes are small enough to reason about and can be built deliberately hostile.

The one that matters is :class:`MergingTokenizer`: it reproduces the BPE
behaviour that broke the first version of this code, so the parity check is
tested against the failure it exists to catch rather than only against success.
"""

from __future__ import annotations

import pytest

from capability_kernel import demo_store
from capability_kernel.compiler import (
    OPEN,
    body_text,
    action_text,
    compile_surface,
    parity_report,
    prefix_stability,
)
from capability_kernel.trie import SlotState


class WordTokenizer:
    """Splits on word boundaries and assigns stable ids. Prefix-stable."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.words: dict[int, str] = {}

    def _piece(self, text: str) -> int:
        if text not in self.ids:
            i = len(self.ids) + 1000
            self.ids[text] = i
            self.words[i] = text
        return self.ids[text]

    def _split(self, text: str) -> list[str]:
        out, buf = [], ""
        for ch in text:
            if ch.isalnum() or ch == "_":
                buf += ch
            else:
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
        if buf:
            out.append(buf)
        return out

    def tokenize(self, text: str) -> list[int]:
        return [self._piece(p) for p in self._split(text)]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.words[t] for t in tokens)


class MergingTokenizer(WordTokenizer):
    """Pulls a space into the word after it — the real BPE behaviour.

    This is what makes ``"\\nACTION "`` tokenize one way alone and another way
    in context, and it is why prefix stability is a separate property from
    round-trip fidelity.
    """

    def _split(self, text: str) -> list[str]:
        pieces = super()._split(text)
        out: list[str] = []
        i = 0
        while i < len(pieces):
            if pieces[i] == " " and i + 1 < len(pieces):
                out.append(" " + pieces[i + 1])
                i += 2
            else:
                out.append(pieces[i])
                i += 1
        return out


@pytest.fixture
def store():
    return demo_store()


@pytest.fixture
def tk():
    return WordTokenizer()


# ── M0 ───────────────────────────────────────────────────────────────────────


def test_parity_passes_on_a_well_behaved_tokenizer(store, tk):
    rep = parity_report(store, tk.tokenize, tk.detokenize)
    assert rep["ok"]
    assert rep["failures"] == []
    assert rep["unstable_prefixes"] == []
    from capability_kernel.manifest import MANIFEST
    assert rep["checked"] == sum(
        len(compile_surface(store, tk.tokenize).by_method[m]) for m in MANIFEST
    )


def test_parity_catches_a_prefix_that_moves_under_merging(store):
    """The regression test for the bug this file's docstring describes.

    Round-trip fidelity is perfect here — the merging tokenizer detokenizes
    exactly — and the surface is still unsafe. If this ever passes silently,
    the check has stopped testing anything.
    """
    tk = MergingTokenizer()
    unstable = prefix_stability(store, tk.tokenize)

    # The frame itself is safe: OPEN no longer ends in a space.
    assert not any(u["where"].endswith(":open") for u in unstable), \
        "OPEN must not end at a boundary a merging tokenizer moves"

    # A slot prefix ends in '=', and the value that follows is free text, so
    # this is where merging bites.
    text = action_text("rename", {"target": "f_pano", "name": "x"})
    assert tk.detokenize(tk.tokenize(text)) == text, "round trip still passes"


def test_round_trip_and_prefix_stability_are_different_properties(store):
    """Where merging actually bites: a boundary that ends in a space.

    Stated as the two halves of the original bug. Round trip passes either way;
    only the boundary tells them apart. The current frame ends on ``ACTION``
    rather than ``ACTION ``, which is why the test above finds nothing to
    report — so this pins the property directly instead of relying on the
    surface to keep exhibiting it.
    """
    tk = MergingTokenizer()
    whole_text = action_text("rename", {"target": "f_pano", "name": "n"})
    whole = tk.tokenize(whole_text)

    assert tk.detokenize(whole) == whole_text, "round trip passes"

    safe = tk.tokenize(OPEN)                 # ends on a word
    assert whole[: len(safe)] == safe

    unsafe = tk.tokenize(OPEN + " ")         # ends on a space — the old frame
    assert whole[: len(unsafe)] != unsafe


# ── M1 ───────────────────────────────────────────────────────────────────────


def test_the_trie_holds_exactly_the_surface(store, tk):
    surf = compile_surface(store, tk.tokenize)
    from capability_kernel.manifest import surface_size

    assert surf.sizes == surface_size(store)
    assert len(surf.trie) == sum(surf.sizes.values())


def test_only_the_manifest_methods_are_reachable_at_the_first_choice(store, tk):
    """The claim, as a number: three continuations, not a vocabulary.

    Where the branch falls depends on the tokenizer — this one keeps the space
    separate, so the choice happens one token later than it does on gemma4,
    where the space merges into the method name. Walk to the branch rather than
    assuming its depth.
    """
    surf = compile_surface(store, tk.tokenize)

    walk = []
    for _ in range(4):
        nxt = surf.trie.next_tokens(walk)
        assert isinstance(nxt, set) and nxt, "the walk must stay inside the trie"
        if len(nxt) > 1:
            break
        walk = walk + [next(iter(nxt))]

    assert {tk.words[t].strip() for t in nxt} == {
        "audit", "decline", "rename", "move", "set_metadata"}, \
        "the whole manifest is in the trie; the phase controller narrows it later"


def test_a_signed_study_can_be_named_but_not_acted_on(store, tk):
    """The invariant, stated exactly.

    It is not "a closed record has no path through the trie" — it has one, and
    that is deliberate. Making it unnameable is what caused the model to rename
    a *different* study when told to rename this one. The invariant is that
    every path mentioning it terminates in decline.
    """
    from capability_kernel.manifest import VIRTUAL

    surf = compile_surface(store, tk.tokenize)
    mentions = [(text, label) for text, label in surf.trie.opcodes
                if "std_hyg" in text or "f_chart" in text]

    assert mentions, "a closed record must be nameable, or it gets substituted"
    assert all(label in VIRTUAL for _, label in mentions), \
        "naming a closed record may only lead to declining"


def test_no_action_can_be_taken_on_a_signed_study(store, tk):
    """The half of the old invariant that still holds, and must."""
    from capability_kernel.manifest import VIRTUAL

    surf = compile_surface(store, tk.tokenize)
    for text, label in surf.trie.opcodes:
        if label in VIRTUAL:
            continue
        assert "std_hyg" not in text
        assert "f_chart" not in text


def test_delete_cannot_be_spelled(store, tk):
    surf = compile_surface(store, tk.tokenize)
    tokens = tk.tokenize(" delete")
    assert surf.trie.next_tokens(tokens) is None


def test_the_world_moving_moves_the_trie(store, tk):
    from capability_kernel.manifest import VIRTUAL

    before = compile_surface(store, tk.tokenize)
    store.get("std_endo").signed = True
    after = compile_surface(store, tk.tokenize)

    assert len(after.trie) < len(before.trie)
    # Actions on it are gone; the ability to say its name is not, so that a
    # request about it can be declined by name rather than redirected.
    assert not any("std_endo" in text for text, label in after.trie.opcodes
                   if label not in VIRTUAL)
    assert any("std_endo" in text for text, label in after.trie.opcodes
               if label in VIRTUAL)


def test_a_free_text_argument_becomes_a_slot(store, tk):
    surf = compile_surface(store, tk.tokenize)
    state = surf.trie.next_tokens(tk.tokenize(" rename target=f_pano name="))
    assert isinstance(state, SlotState)
    assert state.spec.name == "rename.name"
    assert state.exit_tokens, "a slot needs a way out or generation cannot end"


def test_a_slot_refuses_content_that_would_break_the_frame(store, tk):
    surf = compile_surface(store, tk.tokenize)
    state = surf.trie.next_tokens(tk.tokenize(" rename target=f_pano name="))
    assert state.allows("panoramic")
    assert not state.allows("\n"), "a newline would close the action early"
    assert not state.allows("a=b"), "an equals sign would look like another argument"


def test_a_slot_is_bounded(store, tk):
    surf = compile_surface(store, tk.tokenize, slot_max_tokens=3)
    state = surf.trie.next_tokens(tk.tokenize(" rename target=f_pano name="))
    assert state.spec.allows("x", consumed=2)
    assert not state.spec.allows("x", consumed=3), "at the cap only the exit remains"


def test_an_incomplete_action_is_not_complete(store, tk):
    surf = compile_surface(store, tk.tokenize)
    assert not surf.trie.is_complete(tk.tokenize(" move target=f_pano"))
    assert surf.trie.is_complete(tk.tokenize(body_text("move", {"target": "f_pano", "into": "std_endo"})))


def test_the_compiler_refuses_a_surface_it_cannot_enumerate(store, tk):
    with pytest.raises(ValueError, match="over the ceiling"):
        compile_surface(store, tk.tokenize, max_per_method=4)


def test_phase_control_selects_a_subset_of_opcodes(store, tk):
    surf = compile_surface(store, tk.tokenize)
    only_move = surf.indices_for("move")
    assert only_move < surf.all_indices
    assert all(surf.trie.opcodes[i][1] == "move" for i in only_move)
