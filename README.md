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

First draft, private, nothing built yet. **[PLAN.md](PLAN.md)** has the
milestones and — more usefully — the three things already measured that shaped
them:

- **ollama cannot enforce.** It exposes `logprobs`, so it can *observe* mask
  pressure, but no hook to mask the sampler. It is the baseline arm.
- **Gemma 4 is the wrong model** — not because `<|call|>` splits across its
  tokens, though it does, but because every generation opens with
  `<|channel>thought`. Masking from token 0 fights a mandatory reasoning channel.
- **`llama-cpp-python` is the runtime.** Its `LogitsProcessor` is the only place
  that gives enforcement, per-state recomputation, and the rejected mass in one
  hook.

## Scope of the first draft

One patient folder, one level of studies, files and folders both carrying
metadata. Three operations: rename, move, set metadata.

There is no `delete` method, deliberately. The clearest proof of the mechanism is
a capability that structurally does not exist.

## Licence

Apache 2.0. Private until it is worth reading.
