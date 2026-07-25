# Plan — first draft

Build the mechanism from *"Lo que un agente no debería poder hacer"*: a capability
surface compiled from a manifest into a logit mask, recomputed per state, with
the rejected probability mass instrumented.

The exercise domain is deliberately small and real: **a patient's clinical
folder**. One level of folders — each a study — each with metadata, holding files
that also carry metadata. Three operations: rename, move, set metadata.

Small enough to finish. Irreversible enough to matter.

---

## What was measured before writing this

Three findings changed the shape of the plan. All from running the thing, not
from documentation.

**Ollama cannot enforce.** Its OpenAI-compatible endpoint returns `logprobs` with
`top_logprobs`, so it can *observe* how much probability wanted to go somewhere
forbidden — the telemetry in §4.4 of the article. It exposes no hook to mask the
sampler, and its `grammar` option returned an empty response. Ollama is the
baseline arm, not the engine.

**Gemma 4 is the wrong model, and not for the reason expected.** The concern was
that `<|call|>` splits across tokens in its vocabulary, which it does. The real
problem is worse: every generation opens with `<|channel>thought`. It is a
reasoning model whose first tokens are structurally committed to a thinking
channel. Masking from token 0 to a capability surface fights that structure and
produces maximum fallback — syntactically perfect output from a model that has
stopped choosing.

**Use a non-reasoning instruct model.** `Qwen2.5-7B-Instruct` is the pick:
strong tool use, clean structured output, GGUF readily available, comfortable
on-prem. `Llama-3.1-8B-Instruct` and `Ministral-8B` are equivalent fallbacks. The
plan does not depend on which one — but it does depend on the model not having a
mandatory pre-content channel, and that must be checked before anything else.

---

## Architecture

Three pieces from the article, plus one the domain forces.

### 1 · The manifest

A cartridge declares what exists: methods, argument schemas, terminal states.
A file, versionable, signable, reviewable in a PR.

```yaml
cartridge: clinical-folder
version: 0.1.0
methods:
  rename:      { args: { target: entity, name: filename } }
  move:        { args: { target: file,   into: folder } }
  set_metadata:{ args: { target: entity, key: metadata_key, value: string } }
```

### 2 · The compiler

Each legal opcode string is tokenized with the active model's tokenizer and
inserted into a token trie. Given a prefix, the trie answers in O(1) which tokens
may follow. That set becomes the logit mask.

**The compilation step is model-specific**, because it depends on the tokenizer.
A parity check is part of the build, not an afterthought — it is the check that
would have caught the Gemma problem on day one instead of day three.

### 3 · The phase controller — and what the domain adds

The enabled opcode set is a function of state, not a fixed grammar per call.

This domain sharpens that into something the article gestures at but does not
demonstrate: **argument values are enumerated from live state, not just
validated against it.**

The trie does not contain "a filename". It contains *the filenames that
currently exist*. So:

- A file that does not exist **cannot be named.** Not rejected — unnameable.
- A folder that does not exist **cannot be a move destination.**
- A study marked signed **is not in the target set** for rename or move.
- A metadata key outside the value set **cannot be emitted.**
- After a rename, the surface recompiles: the old name is gone from the trie.

That last property is the demo. The mask is not a filter over a static grammar;
it is a projection of the current world, recompiled as the world changes.

### 4 · Mask-pressure telemetry

At each step the processor sees the full distribution before masking. It records
the probability mass assigned to tokens that were then forbidden, per step, per
phase.

This is the piece that does not exist yet anywhere in the codebase. `token-trie`
counts `fellBackSteps` — how often *nothing valid* was in top-K — which is a
different and much weaker signal. What the article describes is the mass, and it
has to be built.

---

## Two arms, because a number needs something to compare against

The claim "100% valid opcodes" means nothing alone. The harness runs the same
task set both ways:

| Arm | Runtime | Mechanism |
|---|---|---|
| **baseline** | ollama | ask in the prompt, parse the reply |
| **enforced** | llama-cpp-python | trie mask in the logits processor |

Same model weights where possible, same tasks, same scoring. Report:

- **invalid-action rate** — baseline will be non-zero, enforced is zero by
  construction, and saying only the second is the failure mode this organisation
  keeps writing about
- **fallback rate** — steps where nothing legal was in top-K and the sampler
  picked without the model's opinion. High fallback means syntactically perfect
  and strategically blind, and it must be published beside the zero
- **task success** — does the constrained arm actually do the job, or does it
  merely fail legally

---

## Milestones

### M0 · Tokenizer parity — *half a day, and it gates everything*

Before any product code. Load the candidate model under `llama-cpp-python`,
tokenize every opcode string the manifest can produce, and assert round-trip
stability. Reject any surface where a legal string does not tokenize the same way
in isolation as in context.

**Done when** a parity report exists for at least two candidate models, and one
is chosen with the reason written down. **If no candidate passes, stop here** —
everything downstream is built on this and finding out later is expensive.

### M1 · Manifest → trie compiler

Port the compiler from
[`token-trie`](https://github.com/EvolvingAgentsLabs/token-trie), which already
compiles enum properties and bounded integers from JSON Schema, and extend it
with **state-enumerated argument types** (`entity`, `file`, `folder`,
`metadata_key`) resolved against a live store at compile time.

**Done when** adding a file to the store and recompiling changes the trie, with a
test asserting the old name is no longer reachable after a rename.

### M2 · Logits processor + phase controller

A `LogitsProcessor` that walks the trie by the tokens generated so far, masks
everything else to `-inf`, and recomputes the enabled set when the phase changes.

**Done when** a scripted session performs rename → move → set-metadata end to
end, and an adversarial prompt instructing the model to delete a file cannot
produce a delete opcode — because `delete` is not in the manifest.

### M3 · Mask-pressure telemetry

Record rejected mass per step, per phase, with the phase and the top rejected
tokens. Emit as structured events.

**Done when** the injection test from M2 shows a visible spike in rejected mass
at the step where the model wanted to comply — which is the article's claim
turned into a chart.

### M4 · The chat

A thin loop over the above: user message → phase → mask → opcode → execute
against the store → result back into context. Terminal UI first; the interface is
not the point.

**Done when** a clinician-shaped conversation ("move the panoramic from March
into the endodontics study and tag it pre-op") completes with every action
structurally legal.

### M5 · The two-arm measurement

Run both arms over a task set. Publish the table with its fallback column.

---

## What is deliberately out of scope for the first draft

**Deletion.** The manifest has no `delete` method. That is the point: the safest
proof of the mechanism is a capability that structurally does not exist, and it
also makes the adversarial test unambiguous.

**Real DICOM, real PHI, real anonymisation.** Synthetic fixtures only. The
mechanism is what is under test, not the pipeline.

**Nested folders.** One level, as specified. Nesting turns the destination
enumeration into a tree walk and buys nothing for the proof.

**Formal verification.** The manifest-to-automaton exporter from §7 of the
article is a later milestone. It is the strongest regulatory argument and it
should be built on a manifest format that has stopped moving.

---

## Honest risks

**The constrained arm may do the job badly.** Constrained decoding distorts the
distribution; that is documented and the article says so. If the enforced arm has
a low invalid-action rate and a low task-success rate, that is the finding, and
it goes in the table next to the zero.

**Enumerating state into the trie has a ceiling.** A folder with ten thousand
files produces ten thousand opcodes and a tokenize call each. `token-trie` caps a
method at 512 for exactly this reason. Beyond that the answer is a trie *slot* —
a node accepting any token from a constrained set and looping — which exists
there for free strings and would need extending. Say the ceiling out loud rather
than discovering it in a demo.

**The model may fight the surface.** Even a non-reasoning model has habits. High
fallback on a well-formed manifest means the surface is unnatural to the model,
and the fix is to reshape the surface toward what the model already emits — not
to push harder on the mask.
