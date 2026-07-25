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

## gemma-4-E4B through transformers

The variant a clinic workstation can run. 3 samples per prompt, 18 turns per arm.

| | unmasked | masked |
|---|---|---|
| **authority** — calls naming a forbidden target or absent capability | **5** | **0** |
| writes to the closed record | 0 | 0 |
| **syntax** — malformed output, invented argument names | 4 | **0** |
| turns producing nothing parseable | 9 | 4 |
| **compliance** — legitimate tasks completed (of 6) | 0 | **3** |
| forbidden requests declined correctly (of 12) | 8 | 7 |
| **cost** — substituted (of 12) | 0 | **3** |

### The security claim is stronger on the better model, not weaker

One authority violation in 30 turns on 12B. **Five in 18 on E4B** — eight times
the rate per turn.

That inverts the obvious reading. 12B did not violate authority because it was
too broken to form a call at all: 14 malformed outputs and 15 turns producing
nothing. A model that cannot name anything cannot name a forbidden thing. E4B is
competent enough to actually try, and it tried five times.

So the security value of enforcement **rises with the capability of the local
model**, up to the point where the model is good enough not to try. Measuring
this on the weakest available model understates it, which is what the 12B table
below does.

## A confound in the compliance column, stated plainly

Both arms emit the same `ACTION method arg=value` text protocol, because that is
what isolates the mask as the only difference. But a real unmasked harness would
use **native tool calling**, which E4B does well — earlier in this project the
same model, through ollama's tool-calling API, executed a move and a metadata
write correctly on the first attempt.

So `0 of 6` understates the unmasked arm. It is not evidence that E4B cannot do
this work; it is evidence that E4B cannot do it *through a text protocol it was
never trained on*. The honest version of the capability claim is narrower:
constrained decoding lets a model emit a protocol it does not know, which is
useful when the protocol is yours and retraining is not an option.

The authority column does not have this problem. Those five calls were formed
well enough to parse and name a forbidden target — the model got far enough to
violate, and did.

## gemma4:12b through llama.cpp

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
