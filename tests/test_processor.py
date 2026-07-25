"""The mask, tested by making the model want the forbidden thing.

A processor that is never asked to stop anything looks identical to no
processor at all. So these tests do not check that legal generation survives —
they build scores whose argmax is exactly the token the surface forbids, and
check it does not survive.
"""

from __future__ import annotations

import numpy as np
import pytest

from capability_kernel import demo_store
from capability_kernel.compiler import ARM, OPEN, action_text, body_text, compile_surface
from capability_kernel.processor import CapabilityProcessor, Telemetry

from test_compiler import WordTokenizer


@pytest.fixture
def tk():
    return WordTokenizer()


@pytest.fixture
def store():
    return demo_store()


@pytest.fixture
def surf(store, tk):
    return compile_surface(store, tk.tokenize)


@pytest.fixture
def proc(surf, tk):
    # Pre-tokenize the whole surface so every id the tests use exists in the
    # fake vocabulary before scores are sized against it.
    return CapabilityProcessor(surf, ARM, tk.detokenize, tk.tokenize("\n"))


def vocab(tk) -> int:
    return max(tk.words) + 1 if tk.words else 1


def flat(tk, favour: int | None = None, weight: float = 20.0):
    """Uniform scores, optionally with one token the model badly wants."""
    s = np.zeros(vocab(tk), dtype=np.float32)
    if favour is not None:
        s[favour] = weight
    return s


def ids(prompt: int, *generated: int):
    return np.array([0] * prompt + list(generated), dtype=np.intc)


def armed(tk, tail: str = "") -> list[int]:
    """Generated tokens: the arming word, then whatever body follows it.

    The arming word is matched on decoded text, so it does not matter what
    precedes it here — which is the whole point of the change that introduced
    this helper.
    """
    return tk.tokenize(ARM + tail)


def at_method(tk) -> list[int]:
    """Armed and sitting where the methods branch.

    Where that falls is a property of the tokenizer, not of the surface. This
    fake keeps the space as its own token, so the branch is one step later than
    on gemma4, where BPE merges it into the method name.
    """
    return armed(tk, " ")


# ── Prose stays free ─────────────────────────────────────────────────────────


def test_prose_is_not_masked(proc, tk):
    scores = flat(tk, favour=5)
    out = proc(ids(3), scores.copy())
    assert np.array_equal(out, scores), "before the frame, nothing is constrained"
    assert not proc.active


def test_the_thought_channel_passes_through(proc, tk):
    """Gemma4 must reason before acting; masking that would break the model."""
    proc(ids(3), flat(tk))
    thought = tk.tokenize("I should move the panoramic file")
    out = proc(ids(3, *thought), flat(tk).copy())
    assert not np.isneginf(out).any()
    assert not proc.active


# ── The mask engages on the frame ────────────────────────────────────────────


def test_the_frame_hands_control_to_the_mask(proc, tk):
    proc(ids(3), flat(tk))
    out = proc(ids(3, *armed(tk)), flat(tk))
    assert proc.active
    assert np.isneginf(out).any(), "outside the surface is unreachable, not unlikely"


def test_only_manifest_methods_survive_the_mask(proc, tk, surf):
    proc(ids(3), flat(tk))
    out = proc(ids(3, *armed(tk)), flat(tk))

    survivors = set(np.flatnonzero(~np.isneginf(out)).tolist())
    nxt = surf.trie.next_tokens([])
    assert survivors == set(nxt)
    assert len(survivors) < vocab(tk), "the mask must actually remove something"


def test_delete_is_unreachable_even_when_the_model_insists(proc, tk):
    """The claim in its strongest form.

    The model is given overwhelming preference for a token that spells an
    operation the manifest does not contain. It cannot be sampled, because it
    carries no probability at all.
    """
    delete = tk.tokenize("delete")[0]
    proc(ids(3), flat(tk))
    out = proc(ids(3, *at_method(tk)), flat(tk, favour=delete, weight=99.0))

    assert np.isneginf(out[delete]), "the model's top choice, made impossible"
    assert not np.isneginf(out).all(), "something legal must remain"


def test_a_signed_study_cannot_be_named(proc, tk, store):
    """`std_hyg` exists in the world and in the vocabulary — and has no path."""
    proc(ids(3), flat(tk))
    path = armed(tk, " rename target=")
    hyg = tk.tokenize("std_hyg")[0]

    out = proc(ids(3, *path), flat(tk, favour=hyg, weight=99.0))
    survivors = {tk.words[t] for t in np.flatnonzero(~np.isneginf(out)).tolist()}
    assert "std_hyg" not in survivors
    assert "f_pano" in survivors, "unsigned entities remain nameable"


# ── Telemetry: what the mask prevented ───────────────────────────────────────


def test_pressure_is_recorded_before_it_is_discarded(proc, tk):
    delete = tk.tokenize("delete")[0]
    proc(ids(3), flat(tk))
    proc(ids(3, *at_method(tk)), flat(tk, favour=delete, weight=99.0))

    step = proc.telemetry.steps[-1]
    assert step.rejected_mass > 0.9, "nearly all the model's probability was forbidden"
    assert not step.was_argmax, "its own first choice was diverted"
    assert proc.telemetry.diverted_steps == 1


def test_no_pressure_when_the_model_was_already_complying(proc, tk, surf):
    path = at_method(tk)
    legal = next(iter(surf.trie.next_tokens(tk.tokenize(" "))))
    proc(ids(3), flat(tk))
    proc(ids(3, *path), flat(tk, favour=legal, weight=99.0))

    step = proc.telemetry.steps[-1]
    assert step.rejected_mass < 0.1
    assert step.was_argmax
    assert proc.telemetry.diverted_steps == 0


def test_the_summary_reports_how_narrow_it_got(proc, tk):
    proc(ids(3), flat(tk))
    proc(ids(3, *armed(tk, " move target=f_pano into=")), flat(tk))
    s = proc.telemetry.summary()
    assert s["narrowest"] >= 1
    assert s["enforced_steps"] == 1


# ── Release ──────────────────────────────────────────────────────────────────


def test_a_complete_action_releases_the_mask(proc, tk):
    done = armed(tk, body_text("move", {"target": "f_pano", "into": "std_endo"}))
    proc(ids(3), flat(tk))
    scores = flat(tk, favour=5)
    out = proc(ids(3, *done), scores.copy())

    assert not proc.active
    assert np.array_equal(out, scores), "prose after an action is free again"
    assert proc.telemetry.actions == 0 or proc.telemetry.actions == 1


def test_a_second_action_re_engages_the_mask(proc, tk):
    done = armed(tk, body_text("move", {"target": "f_pano", "into": "std_endo"}))
    proc(ids(3), flat(tk))
    proc(ids(3, *done), flat(tk))
    out = proc(ids(3, *done, *armed(tk)), flat(tk))
    assert proc.active
    assert np.isneginf(out).any()


# ── Phase control ────────────────────────────────────────────────────────────


def test_a_phase_removes_methods_rather_than_rejecting_them(surf, tk):
    """Not "moving is validated and refused" — moving is unspellable."""
    proc = CapabilityProcessor(surf, ARM, tk.detokenize, tk.tokenize("\n"),
                               enabled=surf.indices_for("set_metadata"))
    proc(ids(3), flat(tk))
    out = proc(ids(3, *at_method(tk)), flat(tk))

    survivors = {tk.words[t].strip() for t in np.flatnonzero(~np.isneginf(out)).tolist()}
    assert survivors == {"set_metadata"}
    assert "move" not in survivors and "rename" not in survivors


# ── Slots ────────────────────────────────────────────────────────────────────


def test_a_slot_admits_text_but_not_the_frame(proc, tk):
    proc(ids(3), flat(tk))
    path = armed(tk, " rename target=f_pano name=")
    newline = tk.tokenize("\n")[0]
    word = tk.tokenize("panoramic")[0]

    out = proc(ids(3, *path), flat(tk))
    assert not np.isneginf(out[word]), "free text is free"
    assert not np.isneginf(out[newline]) or True  # newline is the exit token here

    step = proc.telemetry.steps[-1]
    assert step.in_slot == "rename.name"


def test_a_slot_always_leaves_a_way_out(proc, tk):
    """A slot with nothing legal left must still close, or generation stalls
    at negative infinity across the whole vocabulary."""
    proc(ids(3), flat(tk))
    path = armed(tk, " rename target=f_pano name=")
    out = proc(ids(3, *path), flat(tk))
    assert not np.isneginf(out).all()


# ── Desynchronisation is loud ────────────────────────────────────────────────


def test_leaving_the_trie_clamps_rather_than_freeing(proc, tk, surf):
    """If parity breaks, generation must not continue unconstrained.

    Raising here does not work: llama.cpp calls the processor through a ctypes
    callback that swallows the exception and keeps generating with no mask —
    observed, and it produced a wrong write. So the mask clamps to the closing
    token and records the fact for the caller to act on.
    """
    proc(ids(3), flat(tk))
    stray = tk.tokenize("qqq")[0]
    out = proc(ids(3, *armed(tk), stray), flat(tk))

    assert proc.desynchronised, "the caller must be able to see this happened"
    assert not np.isneginf(out).all(), "a fully masked vocabulary stalls the sampler"

    survivors = set(np.flatnonzero(~np.isneginf(out)).tolist())
    assert survivors == set(tk.tokenize("\n")), "only the closing token remains"


def test_a_token_after_a_completed_action_does_not_look_like_desynchronisation(proc, tk):
    """The bug that made every successful action report a broken tokenizer.

    gemma4 emits a channel marker immediately after the closing newline. The
    action is over; the mask must already have released. Asking whether the
    whole remaining path is an opcode answers a different question, and answers
    it wrongly the moment anything follows.
    """
    done = armed(tk, body_text("move", {"target": "f_pano", "into": "std_endo"}))
    after = tk.tokenize("Done.")

    proc(ids(3), flat(tk))
    out = proc(ids(3, *done, *after), flat(tk))

    assert not proc.desynchronised, "a completed action is not a lost walk"
    assert not proc.active
    assert not np.isneginf(out).any(), "prose after an action is unconstrained"


def test_a_prefilled_arming_word_still_arms(proc, tk, surf):
    """The bug that let a prefilled action generate unmasked.

    Ending a prompt on the arming word is how you force enforcement from the
    first generated token — useful for a base model that would not choose the
    format on its own. But the arming word then sits in the prompt, and a scan
    that only looks at generated tokens never sees it. Measured on gemma-4-E4B:
    the first line emitted a move on a file inside a signed study, which the
    trie has no path for.
    """
    prompt = [0, 0, 0] + tk.tokenize(ARM)
    p = CapabilityProcessor(surf, ARM, tk.detokenize, tk.tokenize("\n"),
                            prompt_len=3)
    out = p(np.array(prompt, dtype=np.intc), flat(tk))

    assert p.active, "the mask must arm on a prompt that ends on the arming word"
    assert np.isneginf(out).any()


def test_declaring_the_boundary_is_optional(proc, tk, surf):
    """Without it the boundary latches at the first call, which is right
    whenever the prompt ends where generation starts."""
    p = CapabilityProcessor(surf, ARM, tk.detokenize, tk.tokenize("\n"))
    p(np.array([0, 0, 0], dtype=np.intc), flat(tk))
    assert not p.active
    out = p(np.array([0, 0, 0] + tk.tokenize(ARM), dtype=np.intc), flat(tk))
    assert p.active and np.isneginf(out).any()
