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

---

# The stronger instruments, and what they found

Both objections above were addressed. Neither changed the answer, and together
they explain why.

`benchmarks/jlens_probe.py`, gemma-4-E4B, calibrated window, 20 scenarios across
four ordinary clerical phrasings. Labels come from the emitted action:
**6 substituted, 9 correct, 5 declined, 0 unparsed.**

## The record signature: zero, everywhere

The instrument that the positive control validates — a fraction of workspace
intensity captured by concept anchors, matched on decoded strings with a prefix
family, so multi-token anchors work.

| signature | substituted | correct | delta |
|---|---|---|---|
| higiene | 0.0000 | 0.0000 | 0.0000 |
| endodoncia | 0.0000 | 0.0000 | 0.0000 |
| ortodoncia | 0.0000 | 0.0000 | 0.0000 |
| cefalometria | 0.0000 | 0.0000 | 0.0000 |

Not "no difference". **Zero.** Not one clinical concept appears in the workspace
of any of the twenty generations.

## The trained probe: chance

Logistic regression over the full workspace, leave-one-out, against 200
permutations.

| | |
|---|---|
| LOO accuracy | 0.600 |
| majority class | **0.600** |
| permutation mean | 0.541 |
| p | 0.299 |

It achieves exactly the majority-class rate, which is what a classifier does
when it learns to always answer "correct". Nothing in the workspace separates
the two.

## Why: the model is not thinking about a clinic

The workspace top during a substituting generation:

    subdirectory · namespaces · dataframe · Namespace · moveTo · relocate
    filesystem · shutil · transposition · rename · transferencia

`shutil` is Python's file-operations module. The model is representing this task
as **generic file manipulation**, and the clinical domain is not present in its
working state at all.

That is a coherent explanation rather than an absence of one, and it accounts for
every null in this file: the per-token probes found nothing because there was
nothing clinical to find; the signature scores zero because it has nothing to
match; the trained probe finds chance because both classes look identical — they
*are* identical, at the level the model is working at.

**It also suggests why the substitution happens.** If the internal state is "move
a file into a folder", then swapping which file is a cheap edit. Nothing in that
representation says one of these is a patient's periodontal chart inside a signed
record. The model is not choosing the wrong clinical entity; it is not
representing clinical entities.

## The consequence, which is a design finding rather than a lens finding

The capability surface names entities `f_chart`, `std_hyg`, `f_pa11` — opaque
identifiers. `store.describe()` shows filenames like `pano_march.dcm`. The whole
vocabulary the model sees while acting is file-shaped, so file-shaped is what it
represents.

**The manifest's naming is not cosmetic. It determines what the model can think
about while it acts.** That is testable: rebuild the surface with semantically
loaded identifiers — `periodontal_chart` inside `hygiene_signed` rather than
`f_chart` inside `std_hyg` — and re-run both instruments. If the clinical
concepts appear in the workspace, the lens question reopens on completely
different footing. If they still do not, the null is about the model.

That experiment costs nothing and has not been run. It is the next one, and it
matters more than model size.

## On bigger models and other families

Not the bottleneck, on this evidence.

- **There is no lens for `gemma-4-12b`**, the variant the mask runs on. The
  family has e2b, e4b and 31b.
- **Qwen separates worse, not better.** sleep-harness measured the same security
  contrast at delta 0.090 on Qwen3.5-4B against **0.141** on gemma-4-E4B. Gemma
  is the stronger substrate for this lens, so switching families would trade
  down.
- **Scale would not address the finding.** A larger model asked to emit
  `move target=f_pa11 into=std_ortho` over a folder of `.dcm` filenames has the
  same reason to represent the task as file manipulation. Spending an A100 on
  `gemma-4-31b` before testing whether the surface's own vocabulary is the cause
  would be answering the second question first.

---

# The naming hypothesis, refuted

The explanation offered above — that the surface's own vocabulary is why the
workspace contains no clinical concepts — was wrong. It was testable, it was
tested, and it failed.

Same folder, same structure, same signed study, same surface size. Only the
identifiers differ: `f_chart` inside `std_hyg`, against `periodontal_chart`
inside `hygiene_study_signed`.

| | opaque | semantic |
|---|---|---|
| substituted | 6 of 20 | **6 of 20** |
| correct | 9 | 9 |
| declined | 5 | 5 |
| every clinical signature | 0.0000 | **0.0000** |
| probe LOO / majority | 0.600 / 0.600 | **0.600 / 0.600** |
| p | 0.299 | 0.517 |

Identical. Not merely similar — the same counts, the same zeros, the same
chance-level probe. Giving the model identifiers that say `periodontal_chart` in
plain words changed neither what it did nor what its workspace contained.

## What that leaves

The model does not represent clinical entities during action emission, and the
names are not why. Two explanations remain, and the second is the more likely.

**The task genuinely is mechanical.** Emitting an action under a mask is
token-level completion inside a trie. The workspace reflects the job in front of
it — `subdirectory`, `filesystem`, `shutil` — because that is the job.

**The readout is in the wrong place.** Every measurement in this file reads the
*action span*. If the model reasons about which record before committing, that
happens in the free prose ahead of the arming word, and the mask deliberately
leaves that segment unconstrained. Reading the emission and concluding the model
never thought about the clinic may be like reading someone's handwriting to find
out what they meant.

That is the next experiment and it is cheap: same rig, readout over the segment
before `ACTION` rather than after. It is not run here.

There is a pattern in this file worth stating plainly: four nulls, and three of
them turned out to be the instrument rather than the model — the layer window,
the multi-token probes, the mismatched scorer. The honest posture is that a
fifth null would still not close the question, and that at some point the cost
of continuing exceeds the value of the answer. For deployment purposes the
question is already settled: **nothing here monitors the substitution failure,
so nothing here licenses autonomous writes.**
