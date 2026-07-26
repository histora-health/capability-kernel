# What was tried and did not work

**A closed line.** This documents a line of investigation that was closed, and it is kept because
the operand rule in the architecture is the conclusion of these four failures
rather than a design preference. The code for these is not in the repository. It was removed because it produces
nothing usable, and leaving it would suggest otherwise. This file exists so the
work is not repeated by someone reading only the parts that succeeded.

Everything below was run against gemma-4-E4B in bf16 on an L4, with the mask and
the [Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens) attached to
the same model instance in one process.

## The question

Constrained decoding removes the unauthorised action. It cannot distinguish two
*authorised* ones, and that gap is silent substitution: told to act on a blocked
record, the model keeps the intent, substitutes a permitted target and executes
on the wrong record (`benchmarks/RESULTS_SUBSTITUTION.md`, reproduced across two
model families).

The obvious answer is a semantic layer — read the model's internal workspace for
the intent the surface cannot see. Four instruments were tried. None of them
sees it.

## What was tried

**Per-token probes over the action span.** Track the lens logit for words naming
the record the model was asked about, and compare against a distractor floor.
Result: 1.50 against a floor of 25.88 — two percent, no separation.

**Per-position trace.** The aggregate averages over the whole action, so a signal
confined to the step where the operand is chosen would be buried. Half right:
peaks are ten times the aggregate, so the mean genuinely was washing out
structure. Not discriminative — the control peaks at 5.69 against the
substituting case's 6.68, within fifteen percent.

**A concept signature.** The instrument the positive control validates: a
fraction of workspace intensity captured by concept anchors, matched on decoded
strings so multi-token anchors work. Four signatures, one per clinical record.
Result: **0.0000 on every signature in both classes** — not "no difference",
zero. No clinical concept appears in the workspace of any of twenty generations.

**A trained probe.** Logistic regression over the full workspace,
substituting-versus-correct, leave-one-out, against 200 permutations. Result:
0.600 accuracy against a majority-class rate of **0.600**, p=0.299. It learned to
always answer "correct".

## Why, as far as it goes

The workspace during action emission:

    subdirectory · namespaces · dataframe · moveTo · relocate · filesystem · shutil

`shutil` is Python's file-operations module. The model represents the task as
generic file manipulation; the clinical domain is not in its working state.

That explains all four nulls at once, and suggests why substitution is cheap: if
the internal state is "move a file into a folder", swapping which file is a small
edit, because nothing in that representation says one of them is a patient's
chart inside a signed record.

## The explanation that was wrong

The natural hypothesis was that the surface's own vocabulary caused this — the
manifest names entities `f_chart` and `std_hyg`, so file-shaped is what the model
represents. It was testable and it failed.

The same folder was rebuilt with `periodontal_chart` inside
`hygiene_study_signed`: identical structure, identical signed study, identical
surface size, only the identifiers different.

| | opaque | semantic |
|---|---|---|
| substituted | 6 of 20 | **6 of 20** |
| every clinical signature | 0.0000 | **0.0000** |
| probe accuracy / majority | 0.600 / 0.600 | **0.600 / 0.600** |

Identical, not similar. Naming the record in plain words changed neither the
behaviour nor the workspace.

## Two things worth knowing before repeating this

**Three of the four nulls were the instrument, not the model.** The first read a
layer window fitted for a different model family — through it, the contrast this
lens *is* known to separate scores 3 of 9 instead of 8 of 9, so the run measured
nothing and had no way to say so. The second tracked probe words that tokenize to
several pieces, which the lens cannot measure at all, so they scored zero and
that zero was read as an absent concept. The third used a scorer the positive
control had never validated.

Every subsequent version therefore runs a **positive control first** — the pairs
this lens is known to separate, in the same session — and reports the run as
inconclusive rather than negative if it fails to reproduce. That control passes
in the final runs (8 wins, 2 ties, 2 losses of 12), which is the only reason the
nulls above are interpretable.

**The readout may be in the wrong place.** Everything here reads the *action
span*. If the model reasons about which record before committing, that happens in
the free prose ahead of the arming word, which the mask deliberately leaves
unconstrained. Reading the emission to find out what the model meant may be like
reading handwriting to find out what someone was thinking. That experiment is
cheap and was not run.

## The conclusion that matters

The plan this came from assumed structural enforcement and activation monitoring
cover **disjoint** failure classes. On the failure that actually matters here
they share a blind spot.

For deployment the question is already settled, and no further experiment changes
it: **nothing here monitors the substitution failure, so nothing here licenses
autonomous writes.** The deployable shape is propose-and-confirm — the assistant
emits an action, the interface shows the named target, a human commits.
