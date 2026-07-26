# capability-kernel

<p align="center">
  <img src="docs/img/capability-kernel.png" alt="Three inputs converge on a shield; three outcomes leave it." width="100%">
</p>

**An open experiment: which small local models can run a clinical records
assistant, and where they break.**

The set of available actions is compiled from live state into a token trie and
applied as a mask on the sampler. At the step where the method name is chosen,
gemma-4-E4B has 3 legal tokens out of 262,144 — and the enabled set is
recomputed every step, so the surface expresses what a JSON Schema structurally
cannot: *only after that*.

A proof of concept, ongoing. What follows is what is measured, including where
it fails.

```
python examples/01_the_surface.py     # what has no path — no model needed
python examples/02_forced_order.py    # 1 reachable opcode of 49, and why
```

## How

A **manifest** declares the methods and where their argument values come from —
a file, versionable, reviewable in a PR. A **compiler** enumerates every legal
opcode *from live state* and tokenizes it into a trie, which becomes the logit
mask. A **phase controller** recomputes which methods have a path at each step.

Argument values are enumerated, not validated. The trie does not contain "a
filename"; it contains the filenames that exist. A signed study is not in any
enum, so nothing can name it — except `decline`, for reasons below.

## What it is actually good for

Two local models, enforcement the only difference
(**[benchmarks/RESULTS.md](benchmarks/RESULTS.md)**):

| | gemma4:12b<br>unmasked → masked | gemma-4-E4B<br>unmasked → masked |
|---|---|---|
| authority violations | 1 → **0** | **5 → 0** |
| writes to the closed record | 0 → 0 | 0 → 0 |
| malformed / invented arguments | 14 → **0** | 4 → **0** |
| legitimate tasks completed | 0 → **10** of 10 | 0 → **3** of 6 |
| substituted — legal write to a record nobody named | 0 → **5** of 20 | 0 → **3** of 12 |

**The security value rises with the model's competence.** One authority violation
in 30 turns on 12B; five in 18 on E4B — eight times the rate. That inverts the
obvious reading: 12B did not violate authority because it was too broken to form
a call at all. A model that cannot name anything cannot name a forbidden thing.
E4B is competent enough to try, and tried five times.

Measuring this on the weakest model available understates it.

**Zero writes reached the closed record in every arm.** A validator with
live-state enums was not defeated here. What enforcement changes is the kind of
guarantee — nothing to catch rather than everything caught — which is
load-bearing when the validator is absent, incomplete, or bypassed, and not
otherwise.

**And the unmasked arms' zero substitutions are not restraint.** They completed
no legitimate work either. Reading that column as prudence is the same mistake as
calling a crashed process secure.

## Ordering, which is the part a schema cannot express

*"This may only happen after that"* is not a statement about shape. Here it is a
statement about which opcodes have a path:

```
store.move(...)          →  pending_audit set
enabled_methods(store)   →  ('audit',)        1 reachable opcode of 49
store.audit('why')       →  the surface reopens
```

A second write is not a call that gets refused; no token beginning one has a
path. A validator enforces the same rule by rejecting — but a rejection is a call
the model produced, which the harness must catch, feed back, and hope is retried
correctly, and the table above shows what retrying costs a small model.

`decline` is unreachable in that phase, and it is the only place it should be: a
state you can decline your way out of permits exactly what the rule prevents.

## What it costs

**Removing the illegal action can produce a wrong one.** Told to act on a record
it cannot touch, the model performs a legal action on a *different* record and
reports success — 5 of 20, and 5 of 5 on the phrasing that sounds like ordinary
filing. Both explicitly adversarial phrasings substituted **0 of 20**
(**[benchmarks/RESULTS_SUBSTITUTION.md](benchmarks/RESULTS_SUBSTITUTION.md)**).

Substitution is triggered by plausibility, not by attack. An attacker crafting an
override is the case everyone tests; a dentist saying *"this was filed in the
wrong place"* is Tuesday.

Two structural causes, both addressed and only one fixed. A surface containing
only ways to act can only be satisfied by acting — hence `decline` as a real
capability. And the model could not name the record it was asked about, so
`decline` alone reads `store.nameable()`: **naming is not acting**, and a closed
record is nameable exactly once, to be declined. That raised correct refusal
attribution from 0.0 to 0.833 and moved the substitution rate not at all.

**The deployable shape is propose-and-confirm, not autonomous writes.** A legal
write to a record nobody named does not appear in a violation log, because
nothing was violated.

## What does not work

**[docs/WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md)** — four instruments
over the model's internal workspace, all null on a rig whose positive control
passes. During action emission the workspace reads `subdirectory · filesystem ·
shutil` and contains no clinical concept, under opaque or semantic identifiers
alike. Structural enforcement and activation monitoring **share** a blind spot
here rather than covering disjoint failure classes.

Three of those four nulls turned out to be the instrument rather than the model.
The file says so, and says what to check first if you repeat it.

## Against published work

**[docs/BADMEMORY.md](docs/BADMEMORY.md)** — read against *Bad Memory*
([arXiv:2607.14611](https://arxiv.org/pdf/2607.14611)), memory-borne injection
against Claude Code and Codex on four frontier models.

| their goal | here |
|---|---|
| credential exfiltration | **unemittable** |
| unauthorised tool use | **unemittable** |
| brand targeting — *"resembles a legitimate user preference"* | **not mitigated**, their highest ASR |

Their threat model is about *trust attribution*. This never attributes trust,
because it never reads intent — a defence that classifies instruction sources
must get the classification right; one that removes the action from the
vocabulary has no classification to get wrong.

Their third goal is our substitution, from the other direction. Different models,
different domain, different mechanism, same shape: **what survives is the request
that looks legitimate.**

## Running it

    pip install -e ".[hf]"                 # transformers — the only way to mask E4B
    pip install -e ".[enforced]"           # llama.cpp — the quantised path
    PYTHONPATH=src:tests pytest tests/ -q  # 64 tests

`gemma-4-E4B` is the variant a clinic workstation can run and llama.cpp will not
load it, reporting 720 of an expected 2131 tensors. `src/capability_kernel/hf.py`
is what makes the deployable model maskable.

    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //')
    B=$B CK_BACKEND=llama PYTHONPATH=src python benchmarks/masked_vs_unmasked.py

## Scope

One patient folder, one level of studies, files and folders carrying metadata.
`rename`, `move`, `set_metadata`, `audit`, `decline`.

There is no `delete`, deliberately: the clearest proof of the mechanism is a
capability that structurally does not exist.

Enumeration has a ceiling. This buys nothing for an agent whose job is arbitrary
execution — a manifest containing `bash` contains everything.

## Licence

Apache 2.0. By [Matias Molinas](https://github.com/matiasmolinas) and
[Ismael Faro](https://github.com/ismaelfaro).
