# Plan — a firmware layer for clinical agents

EAT called it Firmware and implemented it as a string in a system prompt:

```python
self.base_firmware = """
You are an AI agent operating under strict governance rules:
- Never use dangerous imports (os, subprocess, etc.)
"""
```

Governance by asking. This repository is the attempt to make that a compiled
artefact instead, and its first phase measured which mechanisms actually carry
that weight. This plan is what follows from those measurements.

**The goal is not novelty.** It is to adopt the best available approach, add
only what our own results showed to be missing, and validate it on two real
Histora cases well enough to decide whether it ships.

---

## What the first phase established

Four things, all measured, all in `benchmarks/`.

**Post-hoc validation was never defeated.** Zero writes reached a closed record
in any arm, on either model. Whatever else is true, a tool-calling harness with
enums drawn from live state holds the permission line.

**The failure that matters is usability, not authority.** gemma4:12b unmasked
produced 14 malformed outputs in 30 turns and completed 0 of 10 legitimate
tasks. Small models correct their own output badly after a rejection, so
throughput dies on retries that do not converge.

**Constrained decoding fixes that and introduces silent substitution.** 10 of 10
tasks completed, no malformed output — and, when asked to act on a blocked
record, the model kept the intent, substituted a permitted target and executed
on the wrong record. 5 of 20 on one model, 3 of 12 on the other, triggered by
ordinary clerical phrasing and not by adversarial framing.

**Reading the model's internal state does not detect it.** Four instruments,
all null, on a rig whose positive control passes (`docs/WHAT_DID_NOT_WORK.md`).

---

## What we adopt rather than invent

**[AgentSpec](https://cposkitt.github.io/files/publications/agentspec_llm_enforcement_icse26.pdf)'s
rule model** (ICSE 2026), evaluated across code agents, embodied agents and
autonomous driving, with millisecond overhead:

```
rule <id>
  trigger: before an action | on a state change | on task completion
  check:   a conjunction of predicates
  enforce: user inspection | self-reflection | a predefined action
end
```

Three things about it are settled and we will not relitigate them. Enforcement
belongs at the decision point, not at the sampler — our own numbers say the
same. `user inspection` is a first-class enforcement action, so
propose-and-confirm is already in the literature rather than a concession we
invented. And rules are authored, which is fine: a predicate can read state.

**We adopt the shape, not a DSL.** AgentSpec's contribution is a language so
that non-engineers can author rules. For a proof of concept that is premature —
rules will be Python objects with callable predicates, carrying the same three
fields. If the PoC ships, a surface syntax is a later question.

---

## What we add, and the measurement that justifies each

### 1. An option surface derived from state

Not as a security argument — AgentSpec's predicates already cover that. As a
**usability** one, which is where our results found the real problem.

Every rejection costs a retry, and retries are what a small local model cannot
do. So the fewer invalid actions the model can propose, the better it performs,
independently of whether the invalid ones would have been caught.

Already built: `tool_schemas(store)`, `enabled_methods(store)`.

### 2. Operand verification as a rule type

AgentSpec's rules reason about *the action*. This one reasons about **the
relation between the action and the request** — whether the target the model
chose corresponds to anything the user actually named.

It is the only defence that caught silent substitution, and the one rule type
absent from the work reviewed.

Already built: `reference.py`. Becomes a rule rather than a special case.

---

## The two cases

Chosen because they are real for Histora, because they cover the two classes of
constraint, and because they validate different additions.

### Case A — Procedure coding

**Structural constraints.** A clinician dictates; the system proposes the coded
procedure.

    "obturación oclusal en el 36"
    → procedure: restoration
      tooth:     36        (present in this patient's odontogram)
      surface:   occlusal  (a surface tooth 36 has)
      code:      <from the active value set for this payer>

The rules are anatomical and administrative rather than permission-based: a
tooth recorded as absent has no procedures; an incisor has no occlusal surface;
a code must belong to the value set in force for this payer and date.

**What it validates.** State-derived enumeration where it pays — a hallucinated
code is a rejected claim — and chained narrowing, where each step's options are
a function of the previous choice.

**Why this case is favourable.** The clinician asks for anatomically valid
things, so the model is kept inside a vocabulary rather than denied something.
Silent substitution should largely not arise, and if it does anyway that is a
finding.

### Case B — Study ingestion and filing

**Permission constraints, and the live injection surface.** Studies arrive from
clinics whose hygiene we do not control, and DICOM metadata is free text that
the assistant reads.

    incoming study → file into a patient folder
                   → set metadata
                   → request anonymisation
                   → export (only after anonymisation confirms)

**What it validates.** Operand verification against the request, forced ordering
across an irreversible action, and propose-and-confirm under content the
attacker controls.

**Why this case is unfavourable, deliberately.** It is where the first phase
measured every failure. If the additions hold here they hold anywhere.

---

## Milestones

### M0 · Firmware core

`src/capability_kernel/firmware/` — `Rule` with `trigger`, `check`, `enforce`;
a `Runtime` that evaluates rules at the decision point; the three enforcement
actions (`block`, `inspect`, `substitute_action`).

Done when: rules fire in the right order, `inspect` surfaces a proposal instead
of executing, and the existing store operations run through it unchanged.

### M1 · Option surface as a first-class component

Generalise `manifest.py` off the single hard-coded manifest. A domain declares
its methods and where argument values come from; the surface is computed per
turn. Two domains must coexist in one process.

Done when: Case A and Case B manifests both load, and `enabled_methods` is
driven by a per-domain phase function rather than by a hard-coded audit rule.

### M2 · Operand verification as a rule

Move `reference.py` behind the `Rule` interface. Keep the string-overlap
resolver as the default and leave the interface open for the dual-embedding
resolver from `evolving-memory` if it proves necessary.

Done when: the three measured substitutions are blocked through the rule engine
rather than through a special case in the chat loop.

### M3 · Case A end to end

Odontogram manifest, value-set loader, chained phases, dictation → proposal.

Done when: a dictation set produces coded proposals, and the gates below are
measurable.

### M4 · Case B end to end

Ingestion manifest, the anonymisation-before-export ordering, DICOM metadata in
the context as the injection vector.

Done when: the same gates are measurable, with the metadata field carrying
adversarial and plausible content.

### M5 · Validation

Both cases against the gates, on a model that runs on a clinic workstation.

---

## The gates, decided in advance

Three numbers, and the third is the one that usually gets forgotten.

**Coverage.** The fraction of real requests that end in a correct proposed
action without a retry. This is what the first phase showed to be the actual
constraint — 0 of 10 unmasked — and it is what decides whether the product is
usable at all.

**Writes to the wrong operand.** Must be zero. This is what the new rule type
exists to guarantee, and the failure nobody else is measuring.

**Friction.** Human confirmations per session. A system that is safe and asks
for confirmation on everything is not deployable, and finding that out after
building it is the expensive way.

If coverage and operand hold and friction does not, the problem is UX rather
than architecture, and we will know which.

---

## What happens to the existing code

**Kept and evolved.** `store.py` (the domain model), `manifest.py` (becomes the
option surface component), `reference.py` (becomes a rule), `backends.py`, the
benchmarks and their results.

**Moved to `experiments/mask/`.** `compiler.py`, `trie.py`, `processor.py`,
`hf.py` — the sampler enforcement, with the measurements that produced the
substitution finding. Not deleted: the argument for the new architecture is that
the mask did not carry it, and that argument is only checkable if the mask
stays runnable.

**Rewritten.** `chat.py` becomes the agent loop over native tool calling with
the rule engine at the decision point. The examples are rewritten around the two
cases.

The README is rewritten around the firmware framing, with the first phase as the
evidence section rather than the subject.

---

## What this plan does not do

It does not compete with AgentSpec on runtime enforcement, or reimplement
constrained decoding that vLLM and SGLang already ship — including
region-scoped guided decoding, which is the same hybrid design as ours and is
being standardised upstream.

It does not chase the substitution finding into a paper. The finding is a
warning that shapes the architecture; publishing it is a separate decision.

And it does not assume the small model can do this. That is the question, and
the gates are how it gets answered.
