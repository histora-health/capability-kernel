# Masked against unmasked, same model, same runtime

The only difference is whether the logits processor is attached. Same prompts,
same store, same parser, same bounds, temperature 0.7, 5 samples per prompt over
6 prompts — 30 turns per arm.

Reproduce: `B=<gguf> CK_BACKEND=llama PYTHONPATH=src python benchmarks/masked_vs_unmasked.py`

## gemma4:12b through llama.cpp

| | unmasked | masked |
|---|---|---|
| **authority** — calls naming a forbidden target or absent capability | 1 | **0** |
| writes to the closed record | 0 | 0 |
| **syntax** — malformed output, invented argument names | 14 | **0** |
| turns producing nothing parseable | 15 | **0** |
| **compliance** — legitimate tasks completed (of 10) | **0** | **10** |
| forbidden requests declined correctly (of 20) | 15 | 15 |
| **cost** — substituted: legal write to a record nobody named (of 20) | 0 | **5** |

## What this says, in order of how much it matters

### The security advantage is small

One authority violation in 30 unmasked turns, caught by the validator. **Zero
writes to the closed record in either arm.** An unmasked harness with enums
drawn from live state is not defenceless, and here it was not defeated.

The mask's guarantee is different in kind — nothing to catch rather than
everything caught — and that difference is load-bearing only when the validator
is absent, incomplete, or bypassed. It is not the difference between a safe
system and an unsafe one on this evidence.

### The capability advantage is large

**0 of 10 legitimate tasks completed unmasked. 10 of 10 masked.** Fourteen
malformed calls against none, fifteen turns producing nothing parseable against
none.

This is the result worth reporting. The mask did not make a dangerous model
safe; it made an unusable model usable. A 12B model asked to emit structured
actions could not reliably form one, and constraining the sampler removed the
problem entirely.

### The unmasked arm's zero substitutions are not judgement

It substituted nothing because it wrote nothing — 0 legitimate tasks completed,
15 turns with no parseable output. It is inert rather than prudent, and reading
that column as restraint is the same mistake as reading a crashed process as
secure.

So the honest comparison on this model is not *safe* against *unsafe*. It is **a
model that cannot act** against **a model that acts, does the work, and
sometimes acts on the wrong record**.

### The cost is real and it is the reason for propose-and-confirm

Five substitutions in 20 forbidden requests. All from the phrasing that sounds
like ordinary filing — see `RESULTS_SUBSTITUTION.md`, where the same failure is
5 of 5 on that prompt and 0 of 20 on both explicitly adversarial ones.

A legal write to a record nobody named does not appear in a violation log,
because nothing was violated. That is worse in a clinical folder than a rejected
illegal call, which leaves the data untouched.

## Limits

- One model, 30 turns per arm, one folder, six prompts. These are descriptions
  of this setup, not rates.
- `gemma-4-E4B` — the variant a clinic workstation can actually run — needs the
  transformers backend, since llama.cpp reports 720 of an expected 2131 tensors
  for it. That arm is the more interesting one, because E4B follows tool schemas
  competently and so the comparison there measures authority rather than syntax.
- The unmasked arm is given the same phase narrowing and the same live-state
  enums. Anything less would be measuring the manifest rather than the mask.
