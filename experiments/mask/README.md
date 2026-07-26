# The mask — a retired approach, kept runnable

Sampler-level enforcement: the capability surface compiled into a token trie and
applied as a logit mask, so an unauthorised action cannot be emitted at all.

**This is not the architecture.** It is the approach that preceded it, and the
argument for what shipped is that the mask did not carry it. That argument is
only checkable if the mask still runs, so it does.

## What it claimed, and what happened to the claim

**"Unemittable beats validated."** Retired. Post-hoc validation was never
defeated — writes reaching a closed record were zero in every arm measured. What
fails is the *retry* after a rejection, and that is a usability argument, not a
security one.

**"Constrained decoding makes a small model usable."** True and confounded. The
same model went from 0 of 10 legitimate tasks to 10 of 10, but the 0-of-10 arm
used a text protocol rather than native tool calling, so the comparison measures
protocol as well as enforcement. Do not quote the figure without that caveat.

**And it produced a failure of its own.** Forced to choose a valid option, a
model asked about a blocked record keeps the intent and acts on a permitted
record instead — legal action, permitted arguments, wrong record, nothing
logged. That finding is why enforcement moved to the decision point, and it is
what the shipping architecture's operand rule exists to catch.

The trigger is plausibility, not pressure: an explicit fabricated override
produced 0 substitutions in 20 attempts, and "it was filed in the wrong place"
produced 5 of 5.

## Contents

    RESULTS.md                  the two-arm comparison
    RESULTS_SUBSTITUTION.md     which phrasings trigger it
    masked_vs_unmasked.py       the two-arm run
    phrasing.py                 the substitution sweep
    mask_vs_no_mask.py          the same thing as a readable demo

    trie.py compiler.py         the surface as a token trie
    processor.py hf.py          the mask, for llama.cpp and for transformers
    chat.py                     the enforced chat loop
    tests/                      39 tests

`hf.py` exists because gemma-4-E4B — the variant a clinic workstation can run —
will not load in llama.cpp, which reports 720 of an expected 2131 tensors.

## Running it

    pip install -e ".[mask]"
    PYTHONPATH=src:experiments/mask pytest experiments/mask/tests -q

    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //') \
      PYTHONPATH=src:experiments/mask python experiments/mask/masked_vs_unmasked.py

## Also relevant

Constrained decoding is commodity now. vLLM and SGLang compile JSON Schema to a
finite-state machine and mask the vocabulary per step, and there is an open
[RFC in vLLM](https://github.com/vllm-project/vllm/issues/39848) for
*region-scoped guided decoding* — grammar inside the tool-call region, free
generation outside it — which is the same hybrid this code implemented
independently, being standardised upstream. That is the other reason not to
carry it: it is someone else's layer now.
