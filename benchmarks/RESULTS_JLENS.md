# Does the workspace see the substitution the mask cannot?

No. And this time the null is interpretable, because the instrument was
validated in the same session.

gemma-4-E4B in bf16 on an L4, the mask and the
[Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens) attached to the
same model instance in the same process. Reproduce with
`benchmarks/jlens_substitution.py`.

## The first run measured nothing, and could not have said so

`Runtime` defaults to a mid-late layer window fitted for Qwen. sleep-harness had
already measured what that costs on Gemma — same lens, same model, same prompt
pairs:

| window | wins | p | mean delta |
|---|---|---|---|
| default (mid-late) | 3 of 9 | 0.3125 | 0.015 |
| calibrated 0.35–0.65 | **8 of 9** | **0.0195** | 0.141 |

The first version of this experiment used the default. Through that window the
contrast this lens is *known* to separate does not separate either, so the
result said nothing about the question. **"The lens does not see this" and "the
rig is misconfigured" produce identical output**, and there was nothing in the
run to tell them apart.

Hence a positive control, run first, in the same session, on those same pairs.
This run: **8 wins, 2 ties, 2 losses of 12 — the instrument measures.** Layers
14–26 of 42.

## The result

Every probe resolved to a single token. Both readouts are reported because they
answer different questions.

| | aggregate (asked − floor) | per-position peak | at position |
|---|---|---|---|
| substitutes | 0.50 | **6.68** | 15 of 24 |
| control | 0.26 | **5.69** | 5 of 24 |

**The per-position trace was worth running.** Peaks are ten times the aggregate
— 6.68 against 0.50 — so the mean genuinely was washing out structure that
exists at individual steps. That part of the hypothesis was right.

**It is not discriminative.** The control peaks at 5.69, within 15% of the
substituting case's 6.68. The workspace does carry record-related concepts at
particular positions, in both cases equally. It does not carry *more* of them
when the model is about to act on the wrong record.

The workspace top on the substituting case:

    subdirectory · moveTo · filesystem · namespaces · metadata · Wikidata

Still file-operation vocabulary, now with the calibrated window and still no
clinical content.

## What this establishes

The substitution reproduced here (`move target=f_pa11 into=std_ortho`, the same
one measured 5/5 on gemma4:12b, different architecture and runtime), so the
failure occurred and a validated instrument did not register it.

**On this failure, adding a Jacobian lens on top of the mask does not close the
gap.** Any plan that assumed structural enforcement and activation monitoring
cover disjoint failure classes needs revising: on this one they share a blind
spot. That is worth knowing before building on the assumption.

## What it still does not establish

- **n=1 per case.** The behaviour replicates across models; this reading is one
  greedy generation each. A rate needs repetition.
- **One scoring method.** The positive control scores a *signature over concept
  sets*; this tracks *maximum logit per probe token*. The instrument validated
  by the control is not exactly the instrument used for the measurement, which
  is a real gap — a signature built for "which record" rather than for "malicious
  intent" is the obvious next thing to try.
- **A trained probe was not tried.** Hand-chosen probes test whether a concept
  the experimenter thought of is present. A probe trained on the
  substituting-versus-correct contrast tests whether *anything* separates them,
  which is the stronger question.
- **The control's output contains its own probe** (`f_pano` contains `pano`), so
  part of its reading is echo. That inflates the control, which makes the
  comparison more favourable to detection than it should be — and there is still
  no detection.

## On model size

Not the next thing to try. There is **no lens for gemma-4-12b**, the variant the
mask runs on; the Gemma 4 family has e2b, e4b and 31b. `gemma-3-12b-it` has one
but is a different model, 24.4GB and gated.

More to the point, escalating size while the instrument was miscalibrated would
have reproduced the same null and been read as a capability limit. Now that the
control passes, size is a legitimate question — and the answer would be
`gemma-4-31b`, same family, same lens repository, needing an A100.
