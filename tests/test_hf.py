"""The transformers port, tested on a tokenizer that shares nothing with gemma's.

This is not lens infrastructure, whatever its origin. It is the only way to mask
gemma-4-E4B — the variant a clinic workstation can run, and the one llama.cpp
refuses to load, reporting 720 of an expected 2131 tensors. Without this file the
mask works on a model that cannot be deployed and not on the one that can.

`tiny-random-gpt2` is a few megabytes and its vocabulary is unrelated to any
model this targets, so a mask that narrows correctly here is narrowing on the
trie rather than on anything model-specific.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capability_kernel import demo_store
from capability_kernel.compiler import ARM, compile_surface
from capability_kernel.hf import HFCapabilityProcessor

TINY = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TINY)


@pytest.fixture
def proc(tokenizer):
    surface = compile_surface(
        demo_store(), lambda s: tokenizer.encode(s, add_special_tokens=False))
    return HFCapabilityProcessor(surface, tokenizer)


def scores(tokenizer, favour: int | None = None, weight: float = 40.0):
    s = torch.zeros((1, len(tokenizer)))
    if favour is not None:
        s[0, favour] = weight
    return s


def ids(*tokens: int):
    return torch.tensor([list(tokens)], dtype=torch.long)


def test_prose_passes_through_unchanged(proc, tokenizer):
    s = scores(tokenizer, favour=5)
    out = proc(ids(1, 2, 3), s.clone())
    assert torch.equal(out, s)


def test_the_mask_narrows_to_the_trie(proc, tokenizer):
    armed = tokenizer.encode(ARM, add_special_tokens=False)
    proc(ids(1, 2, 3), scores(tokenizer))
    out = proc(ids(1, 2, 3, *armed), scores(tokenizer))

    survivors = torch.nonzero(~torch.isinf(out[0])).flatten().tolist()
    assert 0 < len(survivors) < len(tokenizer)
    assert {tokenizer.decode([t]).strip() for t in survivors} == {
        "re", "dec", "mov", "set"}, "the four methods, however this tokenizer splits them"


def test_a_forbidden_top_choice_is_removed(proc, tokenizer):
    """The claim, on a tokenizer that has never seen this manifest."""
    delete = tokenizer.encode("delete", add_special_tokens=False)[0]
    proc(ids(1, 2, 3), scores(tokenizer))
    armed = tokenizer.encode(ARM, add_special_tokens=False)
    out = proc(ids(1, 2, 3, *armed), scores(tokenizer, favour=delete))

    assert torch.isinf(out[0, delete]) and out[0, delete] < 0
    assert not torch.isinf(out[0]).all(), "something legal must remain"


def test_dtype_and_device_survive(proc, tokenizer):
    armed = tokenizer.encode(ARM, add_special_tokens=False)
    proc(ids(1, 2, 3), scores(tokenizer))
    s = scores(tokenizer).to(torch.float16)
    out = proc(ids(1, 2, 3, *armed), s)
    assert out.dtype == s.dtype and out.shape == s.shape


def test_a_batch_is_refused(proc, tokenizer):
    """Masking a batch against one sequence's walk would constrain every other
    row to the wrong step — output that looks fine and was enforced against the
    wrong state."""
    with pytest.raises(ValueError, match="batch size"):
        proc(torch.zeros((2, 3), dtype=torch.long), torch.zeros((2, len(tokenizer))))


def test_a_prefilled_arming_word_arms(tokenizer):
    """Ending a prompt on the arming word forces enforcement from the first
    generated token, which is what makes a base model usable."""
    surface = compile_surface(
        demo_store(), lambda s: tokenizer.encode(s, add_special_tokens=False))
    armed = tokenizer.encode(ARM, add_special_tokens=False)
    proc = HFCapabilityProcessor(surface, tokenizer, prompt_len=3)

    out = proc(ids(1, 2, 3, *armed), scores(tokenizer))
    assert torch.isinf(out[0]).any()
