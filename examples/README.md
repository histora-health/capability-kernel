# Examples

Three, in the order of the architecture they demonstrate. The first two need no
model and take a second; the third needs a gguf and a few minutes.

**`01_the_surface.py`** — layer one, an option surface derived from live state.
What has no path, and why a signed record stops being a possible value the
moment it is signed. No model.

**`02_forced_order.py`** — layer two, order forced at the system level. While a
change is unrecorded, one opcode of forty-nine has a path and it is the one that
records it. No model.

**`03_with_a_model.py`** — what a real model does with and without the mask,
including silent substitution: asked to act on a blocked record, it keeps the
intent and acts on a permitted one instead. Needs a model.

## Why two of them need no model

Because what they demonstrate is a property of a compiled artefact, not a
statistic over samples. "This action has no path through this trie" is
verifiable by inspecting the surface; if showing it required generating a
thousand samples and counting, it would be statistical evidence again.

**They demonstrate nothing about whether a model complies.** That is `03`, and
`03` is also where the mechanism's cost appears. Read it before believing the
first two.

## Running them

    pip install -e ".[hf]"
    python examples/01_the_surface.py
    python examples/02_forced_order.py

    pip install -e ".[enforced]"
    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //') \
      PYTHONPATH=src python examples/03_with_a_model.py

`benchmarks/masked_vs_unmasked.py` is `03` at n=5 with the full scorecard —
authority, syntax and compliance counted separately, plus the substitution cost.
`benchmarks/phrasing.py` measures which wordings trigger substitution.
