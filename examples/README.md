# Examples

Three, in order. The first two need no model and take a second; the third needs
a gguf and a few minutes.

| | shows | model |
|---|---|---|
| `01_the_surface.py` | what is unreachable, and why that is inspectable | no |
| `02_forced_order.py` | ordering as a constraint a schema cannot express | no |
| `03_with_a_model.py` | what a model actually does, both ways | **yes** |

## Why two of them need no model

Because the claim they demonstrate is about an artefact, not about behaviour.
"`delete` has no path through this trie" is verifiable by inspecting the
compiled surface. If showing it required generating a thousand samples and
counting, it would be statistical evidence again — which is the thing this
mechanism exists to replace.

That is the whole of the auditability argument: a log shows a system did not do
something; a compiled manifest shows it could not.

**They demonstrate nothing about whether a model complies.** That is `03`, and
`03` is also where the mechanism's cost shows up. Read it before believing the
first two.

## Running them

    pip install -e ".[hf]"
    python examples/01_the_surface.py
    python examples/02_forced_order.py

    pip install -e ".[enforced]"
    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //') \
      PYTHONPATH=src python examples/03_with_a_model.py

`benchmarks/masked_vs_unmasked.py` is `03` at n=5 with the full scorecard —
authority, syntax, compliance, and the substitution cost, counted separately
because conflating them is how this gets oversold.
