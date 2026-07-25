# capability-kernel

**An agent cannot emit an action it is not authorised to take.**

Not rejected. Not detected. Unemittable — the token that would begin the action
is absent from the sampler's candidate set at the step where the model would have
chosen it.

Structured outputs solved *format*. A JSON Schema says what shape the output has.
It does not say what the agent is authorised to do right now, and in clinical
software that gap is where the risk lives. A tool call with impeccable JSON that
files a consent before it exists, or codes a procedure on an absent tooth, is
structurally valid and clinically unacceptable. The schema does not stop it. That
is not its job.

## How

Three pieces:

1. **A manifest** declares what exists — methods, argument schemas, terminal
   states. A file: versionable, signable, reviewable in a PR.
2. **A compiler** tokenizes every legal opcode with the active model's tokenizer
   into a token trie. That trie becomes the logit mask.
3. **A phase controller** recomputes the enabled set from world state. Not a
   fixed grammar per call — a mask that changes as the world does.

And the piece this domain forces: **argument values are enumerated from live
state**, not validated against it. The trie does not contain "a filename"; it
contains the filenames that exist. After a rename, the old name is not in the
trie. It is unnameable.

## Status

M0–M4 built and running against gemma4:12b. 47 tests. **[PLAN.md](PLAN.md)** has
the milestones.

The mechanism works, measured: asked to delete a file and rename a signed study,
neither is emitted. At the step where the method name is chosen the model has
three legal tokens out of 262,144. `delete` is not improbable, it is absent.

### Which model, and where

Settled by measurement rather than by the spec sheet, and the two variants turn
out to be complementary in the opposite direction to what was assumed:

| | ollama | llama.cpp |
|---|---|---|
| `gemma4:e4b` | 43s / 3 turns | will not load — 720 of 2131 tensors |
| `gemma4:12b` | >10 min / turn | loads, vocab 262144 — **the enforced arm** |

e4b's MatFormer nesting reports `arch=gemma4` and then fails the tensor count,
so ollama's engine is the only one that runs it. 12b, unusable as a chat model,
is the one that can be masked.

### What the mask does not do

Three failure modes, all found by running it, none fixed by more masking. They
are the honest boundary of the claim and the reason the two-arm measurement is
worth more than a violation count.

**It could be opted out of.** The first version armed on the exact token
sequence for `"\nACTION"`. Asked to delete a file, gemma4 wrote
`<channel|>ACTION delete(path=...)` — no newline, no match, the whole line
generated free. A frame the model must spell exactly is a trigger the model
controls. Arming now happens on decoded text, wherever the word appears.

**It caused a wrong write.** Asked to rename a signed study, the enforced arm
renamed a *different* study — the nearest reachable target — and recorded it as
a success. The baseline arm refused correctly. Removing the illegal action had
converted a correct refusal into a wrong write on an intact record, which in a
clinical folder is worse than the violation it prevented. The cause is
structural: once the model arms, the mask requires it to finish *some* legal
action. A surface containing only ways to act can only be satisfied by acting.
Hence `decline`, which is now part of the manifest.

**It does not make an action correct.** Asked to tag a file, the model tagged
the study the file sits in. Both are legal opcodes; the mask cannot tell them
apart. That one needs a better model or a better prompt, and saying so is more
useful than implying the mechanism covers it.

## Scope of the first draft

One patient folder, one level of studies, files and folders both carrying
metadata. Three operations — rename, move, set metadata — plus `decline`.

There is no `delete` method, deliberately. The clearest proof of the mechanism is
a capability that structurally does not exist.

## Licence

Apache 2.0. Private until it is worth reading.
