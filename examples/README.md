# Examples

Three, in the order of the architecture they demonstrate. The first two need no
model and run instantly; the third needs a gguf and about a minute.

**`01_the_surface.py`** — block one, an option surface computed from live state.
What has no representation, and why a signed record stops being a possible value
the moment it is signed. No model.

**`02_forced_order.py`** — block two, order forced at the system level. While a
change is unrecorded, one method is available and it is the one that records it.
No model.

**`03_the_agent.py`** — all of it on a real model: propose, verdict, confirm.
Three requests, one per block. Needs a model.

## Why two of them need no model

Because what they demonstrate is a property of a computed artefact, not a
statistic over samples. "This action is not in the offered set" is verifiable by
inspecting the surface; if showing it required generating a thousand samples and
counting, it would be statistical evidence again.

**They demonstrate nothing about whether a model complies.** That is `03`, and
`03` is one sample — read it as an illustration and
[`benchmarks/`](../benchmarks/) as the evidence.

## What `03` looked like when this was written

All three requests behaved. The two that historically produced silent
substitution — renaming a signed study, and the "it was filed in the wrong
place" phrasing that produced it 5 times out of 5 — both came back as declines
naming the record that was actually asked about.

Worth reading precisely: the **surface** handled both, so the operand rule never
had to fire. That rule is the backstop for when the surface cannot help, which
is why its false-positive rate is measured separately in
[`benchmarks/RESULTS_VALIDATION.md`](../benchmarks/RESULTS_VALIDATION.md) rather
than inferred from runs where it stayed quiet.

## Running them

    PYTHONPATH=src python examples/01_the_surface.py
    PYTHONPATH=src python examples/02_forced_order.py

    pip install -e ".[local]"
    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //') \
      PYTHONPATH=src python examples/03_the_agent.py
