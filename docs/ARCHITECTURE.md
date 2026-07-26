# Architecture

A living document. It describes what this is, what it borrows, what it adds,
every block it is made of, and how it is used. It is updated as the project
moves, and the changelog at the end records what changed and why.

---

## 1. The problem

A clinical records assistant that runs on a small local model inside the
customer's perimeter, because the data cannot leave it.

That constraint changes the objective. With a frontier model over an API the
question is what the model knows; with a small local model the question is what
you can guarantee about what it emits. Both matter, but only the second is
solvable by architecture.

The failure that matters is not the one people expect. It is not the agent
deleting a record — a validator catches that. It is the agent doing something
*legal* that nobody asked for, on a record nobody named, and reporting success.

---

## 2. What EAT called this, and why it did not work

The [Evolving Agents Toolkit](https://github.com/EvolvingAgentsLabs/evolving-agents)
had a governance layer named **Firmware**. Its implementation:

```python
self.base_firmware = """
You are an AI agent operating under strict governance rules:
- Never use dangerous imports (os, subprocess, etc.)
"""
```

A string in a system prompt. The decomposition was right — a system-level
control layer distinct from the agent's reasoning is the correct shape — and the
substrate did not exist. This repository is the attempt to give it one.

---

## 3. State of the art

The field moved substantially through 2026. Four lines are active, and we sit
inside them rather than beside them.

### Runtime enforcement with a rule language

**[AgentSpec](https://cposkitt.github.io/files/publications/agentspec_llm_enforcement_icse26.pdf)**
(ICSE 2026) — a domain-specific language for runtime constraints:

```
rule <id>
  trigger: before an action | on a state change | on task completion
  check:   a conjunction of predicates
  enforce: user inspection | self-reflection | a predefined action
end
```

Evaluated across code agents, embodied agents and autonomous driving. Prevents
over 90% of unsafe executions in code agents, eliminates all hazardous actions
in embodied tasks, with millisecond overhead. Rules are authored or
LLM-generated for human review; predicates can read state; enforcement hooks
into the agent's decision pipeline before execution.

**This is the closest published work to what we need, and we adopt its model.**

### Policy enforcement at the harness and OS level

**[ActPlane](https://arxiv.org/pdf/2606.25189)** — programmable OS-level policy
enforcement for agent harnesses. Complementary rather than competing: it governs
what the harness process may do, we govern what the agent may propose.

### Capability systems for agents

**[Lingering Authority](https://arxiv.org/pdf/2606.22504)** — revocable
resource-and-effect capabilities for coding agents. The classical capability
model applied to agents, with the emphasis on revocation.

### Constrained decoding, now commodity

vLLM and SGLang compile JSON Schema into finite-state machines and apply a
vocabulary mask at each decoding step. There is an open
[RFC in vLLM](https://github.com/vllm-project/vllm/issues/39848) for
*region-scoped guided decoding* — grammar applied only inside the tool-call
region, free generation outside it — which is the same hybrid design this
repository implemented independently, being standardised upstream.

**[Constraint Tax in Open-Weight LLMs](https://arxiv.org/html/2606.25605v1)**
measures what that costs: under simultaneous schema constraints and tool
calling, open-weight models stop invoking tools entirely, with a five-mode
taxonomy of suppression behaviours.

### Independent corroboration of the failure

**[Bad Memory](https://arxiv.org/pdf/2607.14611)** (Gadgil et al., University of
Washington) evaluated memory-borne prompt injection against Claude Code and
OpenAI Codex across Haiku 4.5, Opus 4.7, GPT-5.2 and GPT-5.5. Three planted
goals: credential exfiltration, unauthorised tool use, and brand promotion.

The first two are the overtly malicious class. The third is the one that
matters here — in their words it *"resembles a legitimate user preference"*,
giving the agent the least signal to distinguish a stored preference from a
planted directive — **and it produced their highest attack success rate**.

Different models, different domain, different mechanism, and the same shape as
the failure measured in §5: what survives every defence is the request that
looks legitimate. Their recommendation includes policy tiers so low-trust
content cannot override behavioural constraints, which is the layer this
document describes.

### And the industrial frame

Anthropic's Zero Trust for AI agents, DeepMind's AI Control Roadmap, and the
NIST NCCoE concept paper (February 2026) on agent identity and authorization.

---

## 4. What we adopt, and what we add

### Adopted

**AgentSpec's rule model**, in shape rather than syntax. `trigger`, `check`,
`enforce`, evaluated at the decision point. Rules are Python objects with
callable predicates — a surface language exists so non-engineers can author
rules, which is premature for a proof of concept.

**`user inspection` as a first-class enforcement action.** Propose-and-confirm
is in the literature; it is not a concession we invented because the model was
unreliable.

**Enforcement at the decision point rather than the sampler.** Our own
measurements agree: post-hoc validation was never defeated in any arm.

**Constrained decoding stays upstream.** We do not maintain a mask when vLLM and
SGLang ship one.

### The design law

From [`token-trie`](https://github.com/EvolvingAgentsLabs/token-trie)'s
`CLAUDE.md` §4.2, validated there on a 350M model playing Tetris:

> The LLM-CPU is a ratifier, not a planner. If you find yourself making the LLM
> decide something the Program could compute, push it down into the Program.

**One model call per proposed action.** This is a hard constraint, not an
optimisation. It is what keeps latency inside budget, and it is the same law
that removes the substitution failure — a model that never chooses an operand
cannot choose the wrong one.

### Added, each because a measurement demanded it

**An option surface derived from live state.** Argued on usability, not
security: AgentSpec's predicates already cover the security case. Every
rejection costs a retry, and retries are what a small model cannot do — 14
malformed outputs in 30 turns, 0 of 10 tasks completed.

**Operand verification as a rule type.** AgentSpec's rules reason about the
action; this one reasons about the relation between the action and the request.
It is the only defence that caught silent substitution, and it is absent from
the work reviewed.

---

## 5. What the first phase measured

All of it is in `benchmarks/`, with raw data.

| | |
|---|---|
| Writes reaching a closed record, all four arms | **0** |
| Legitimate tasks completed, unmasked / masked | **0 of 10 / 10 of 10** |
| Malformed outputs, unmasked / masked | 14 / 0 |
| Silent substitution, masked | 5 of 20 and 3 of 12 |
| Substitution under adversarial phrasing | **0 of 20** |
| Substitution under ordinary clerical phrasing | **5 of 5** |
| Latency, gemma4:12b via llama.cpp, warm | ~2s per turn |

Four instruments over the model's internal state failed to detect the
substitution, on a rig whose positive control passes
(`docs/WHAT_DID_NOT_WORK.md`).

**Silent substitution** is what we call the failure: constrained decoding forces
the model to always choose a valid option, so when the requested record is
blocked it keeps the intent, substitutes a permitted target, and executes on the
wrong record. No schema or validator detects it, because the action is
technically legal and every argument is in the permitted vocabulary.

---

## 6. The blocks

```
    user request
         │
    ┌────▼─────────────────────────────────────────┐
    │  Domain            manifest + store          │  what exists, what may be done
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │  Option surface    tool_schemas(store)       │  the tools for THIS turn
    │                    enabled_methods(store)    │  the methods for THIS phase
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │  Model             one call, native tools    │  chooses; never plans
    └────┬─────────────────────────────────────────┘
         │  proposed action
    ┌────▼─────────────────────────────────────────┐
    │  Firmware          Rule(trigger,check,enforce)│  block · inspect · substitute
    │    ├── authority   predicates over state     │
    │    ├── ordering    phase predicates          │
    │    └── operand     reference.check(...)      │  ← the addition
    └────┬─────────────────────────────────────────┘
         │  block → refused    inspect → proposal
    ┌────▼─────────────────────────────────────────┐
    │  Interface         propose-and-confirm       │  a person sees the named target
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │  Store             executes, journals, audits│
    └──────────────────────────────────────────────┘
```

### Domain — `store.py`, `manifest.py`

The world and what may be done to it. A manifest declares methods and where
each argument's legal values come from — a callable over the store, a fixed
vocabulary, or free text.

Values are **enumerated, not validated**. The surface does not contain "a
filename"; it contains the filenames that exist.

### Option surface — `domain.py`

A `Domain` is a value: its methods, where each argument's legal values come
from, and a phase function saying which methods exist right now. Everything the
surface does is a method on it, computed from a store passed in — which is what
lets two domains share one process and keeps `Domain` from assuming the clinical
store's shape.

Regenerated every turn. A signed record leaves the enums the moment it is
signed. This is what shrinks what the model can propose, which reduces
rejections, which is what a small model cannot recover from.

**Chained arguments.** An argument source takes `(store)` or `(store, chosen)`,
where `chosen` is what has been decided already. Which surfaces a tooth has
depends on the tooth; which codes are valid depends on the surface.

That dependency has a consequence for how a chained method reaches the model. A
JSON Schema cannot express it, so computing each enum independently would offer
an occlusal surface on an incisor. **A chained method is therefore offered as a
single enumerated choice over the combinations that exist** — the program
computes them, the model picks one, and the result is valid by construction in
one call rather than three. Three round trips per coded procedure is six seconds
at the measured ~2s per turn, for something a dentist does several times per
consultation.

`MAX_COMBINATIONS` refuses loudly above 400. Enumeration has a ceiling, and a
schema with ten thousand enum values is a worse failure than a refused one.

### Model — one call, native tool calling

Chooses among enumerated options. It does not plan, and it does not construct
operands the program could have computed.

Backends behind one interface (`backends.py`): llama.cpp for quantised local
weights, transformers for models llama.cpp cannot load — `gemma-4-E4B` reports
720 of an expected 2131 tensors under llama.cpp, and it is the variant a clinic
workstation runs.

### Firmware — `firmware/` *(M0, done)*

`Rule(trigger, check, enforce)` evaluated at the decision point, before
execution. Three enforcement actions:

- **block** — the action does not happen; the reason is returned
- **inspect** — the action is proposed to a person rather than executed
- **substitute** — a predefined action replaces the proposed one

Three rule families:

- **authority** — predicates over state; a signed record may not be modified
- **ordering** — phase predicates; export only after anonymisation confirms
- **operand** — the addition; the target must correspond to what was asked

### Operand verification — `firmware/operand.py`, `resolvers.py`

String overlap between the entity's name and the words of the request. It
suffices because someone asking for "the perio chart" uses the words in
`perio_chart.pdf`.

Four details, each found by a failing test rather than by reasoning: name and id
are scored separately, because pooling let `std` and `hyg` dilute a study called
*Hygiene* below threshold; words naming more than one entity are ignored,
because `std_` made every study a candidate; matching is by prefix from four
characters, because clinicians write "panoramic" and the file is
`pano_march.dcm`; and the referent is plural, because a move names what to move
*and* where.

It **reports rather than corrects**. A guard that corrected silently would be
making clinical decisions by string similarity.

**The resolver is a separate thing from the rule**, because the rule's logic is
settled and the resolver's is not. Resolvers work over `(id, name)` pairs rather
than store objects, so one written for the clinical domain works on the
odontogram — nothing in the rule knows about teeth, and nothing in `Odontogram`
knows about resolvers.

A fifth detail, found by running the rule on the second domain: **numeric
identifiers carry reference**. FDI names a tooth in two digits, and dropping
short tokens as noise dropped the only word in a dictation that identified the
record — so the rule allowed a procedure on any tooth at all.

When paraphrase or cross-language reference breaks it — Spanish dictation
against English filenames is the likely case — the swap is
[EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
via sentence-transformers, or the dual-embedding resolver in `evolving-memory`
which indexes each component twice: once for what it *is*, once for what it is
*for*. Neither is written yet, deliberately — a resolver added before it is
needed is one whose failure modes nobody has seen.

### Interface — propose-and-confirm

The assistant structures and proposes; a person sees the **named** target and
confirms. Not caution in general — the specific case where it is needed was
measured.

### Experiment — `experiments/mask/`

Sampler-level enforcement, kept runnable with its measurements. The argument for
this architecture is that the mask did not carry it, and that argument is only
checkable if the mask still runs.

---

## 7. How it is used

```python
from capability_kernel import demo_store
from capability_kernel.firmware import Runtime, Rule

store = demo_store()
runtime = Runtime(store, rules=[
    Rule("closed_record",  trigger="before_action",
         check=lambda a, s, _: s.get(a.target) and s.is_closed(a.target),
         enforce="block"),
    Rule("audit_first",    trigger="before_action",
         check=lambda a, s, _: s.pending_audit and a.method != "audit",
         enforce="block"),
    Rule("operand_matches", trigger="before_action",
         check=lambda a, s, ctx: reference.check(s, ctx.request, a.target),
         enforce="inspect"),
])

proposal = runtime.propose(request="mové la ficha periodontal a ortodoncia")
# → Proposal(action=..., verdict="inspect", reason="the request refers to …")
# the interface shows the named target; a person confirms
runtime.commit(proposal)
```

That is the shape; the working call is `Runtime.propose(action, context)`
returning a `Proposal`, and `Runtime.commit(proposal)` to execute one a person
approved. `Runtime.run` does both for callers with nobody to ask, and refuses on
an inspection rather than guessing.

Two properties worth knowing before building on it. **A proposal executes
nothing**, including when every rule passes — a caller that wants autonomy has
to ask by calling `commit`, which is the right way round. And **commit
re-evaluates**: between proposal and confirmation another user may have signed
the study being renamed, and approving a proposal approves *that* action rather
than a licence to run it against a state nobody saw.

---

## 8. Why this is a production option

**Latency.** gemma4:12b through llama.cpp answers in ~2s per turn warm, with a
14.6s one-time load. The one-call constraint keeps a proposed action at one
round trip, so that number is the budget rather than a fraction of it.

**Cost.** Zero marginal. A Q4 12B is about 8GB and runs on a consulting-room
workstation. No API, no per-token billing, no data leaving the perimeter.

**Model options.** gemma4:12b local is the candidate. `gemma-4-E4B` is the
alternative if coverage falls short — more capable at native tool calling, same
machine, transformers backend. A 26B or 31B in the customer's own cloud is worth
measuring as a ceiling for coverage, not as a deployment path.

**Failure behaviour.** Every failure mode is either blocked, proposed for
inspection, or logged. The one that used to be silent has a rule.

---

## 9. What is not validated

Stated here so it is not discovered late.

**Coverage — answered.** 0.80 for coding with the gap entirely attributable to
value-set structure, 0.944 for ingestion. Native tool calling on a local model
works; the text protocol was what did not.

**Friction.** Filing a study competes with dragging a file, which is already
fast. An assistant that parses a sentence and waits for confirmation may be
slower than what it replaces. Procedure coding has the better product argument
— dictating beats navigating a coding tree — which is why it leads.

**Operand verification across languages.** Zero false positives on twenty-five
requests including Spanish — but the entity names are English and the overlap
works because clinical vocabulary is cognate. "la ficha de encías" against
`perio_chart.pdf` shares nothing, and that is the case EmbeddingGemma exists
for.

**Enumeration at scale.** A demo folder has seven entities. A real clinical
history has hundreds, and the ceiling has been named but not measured.

---

## 10. Changelog

**2026-07-26** — M5 done, and two of the gates turned out to be measuring the
wrong thing.

**Friction was defined as inspections and both cases scored 0.0.** That number
is true and describes nothing a user experiences: under propose-and-confirm
every proposal needs a person, so friction is one confirmation per action and
adjudications are the rare extra on top. Restating it is what decides Case B —
ten seconds plus a confirmation loses to dragging a file.

**And 0.0 inspections also meant the operand rule never fired in either run**,
so the friction number was hiding a guard that had not been exercised.
`benchmarks/validation.py` measures it directly: zero false positives across
twenty-five legitimate requests in three domains and two languages, and three of
three measured substitutions still caught. That false-positive rate is the
number that decides whether a guard survives production, because one that blocks
real work gets switched off.

Verdict: Case A ships after keying the value set by material as well as surface,
which closes the whole coverage gap. Case B does not ship and was never the
product case; it validated the orderings under a planted override and the
operand rule on a second domain, which is what it was built for.

**2026-07-26** — M4 done. `ingestion.py` is Case B, with the DICOM note as the
injection vector and two orderings the option surface enforces by absence:
export does not exist until something is anonymised, and annotation does not
exist until a study is filed.

Coverage 0.944, zero exports of un-anonymised data, zero writes to the signed
study, under every note. **The adversarial note did nothing** — a fabricated
administrative override scored identically to the empty control, and the model
declined by citing the surface back at it.

**The plausible note produced a failure the gates do not catch.** Told by the
user to file into orthodontics, with a sender's note saying the study belongs
with hygiene, the model refused the user. Not privilege escalation — hygiene was
never an available destination — but denial of service on a legitimate request,
achieved through attacker-controlled content, and invisible to every gate
because nothing was written and the refusal reads as diligence. See
`benchmarks/RESULTS_INGESTION.md` for why a prompt-level boundary is
proportionate here and would not be for an action.

Two benchmark errors corrected during the run, both mine. The first expected
`annotate` on an unfiled study and scored the model wrong three times for
correctly saying it had to be filed first. The second read breaches from the
journal, which the setup clears, so three correct exports looked like ordering
violations — a breach is a property of the world, so the check now asks the
world.

**2026-07-26** — M3 done, and the gates say something more useful than a pass.
`agent.py` is the production loop: one model call, native tool calling, firmware
at the decision point, nothing executed. `benchmarks/coding.py` measures it.

Coverage 0.8, wrong operand 0, friction 0.0, latency median 3.48s and p95 8.12s.
The operand gate holds across twenty proposals in a language the value set is
not written in. Structural constraints hold without a rule stating them — a
missing tooth and an occlusal surface on an incisor were both declined with the
reason named, and neither is a prohibition the model was told about.

**Four of twenty carry the wrong code.** Right tooth, right surface, wrong
procedure: "amalgama" coded as composite, with the amalgam code sitting in the
offered set. That is the substitution failure one level down — a legal option
chosen for the wrong reason, on the code rather than the record — and it is what
the case exists to sell against, since a wrong code is a rejected claim.

The first version of the benchmark scored coverage at 1.0 because it checked
only the tooth. Measuring the guard that was built rather than the product that
was promised is the specific way this project has been wrong before.

Two integration facts worth recording. **Gemma 4 emits a tool call in its own
format** — `<|tool_call>call:name{arg:<|"|>value<|"|>}` — which llama.cpp does
not normalise, so a correct call arrives as content and the turn reads as a
refusal. `backends.parse_tool_calls` recovers it. And a chained method reaches
the model as one `choice`, so the label it returns is expanded back into
arguments against the enumerated set — a label not in that set means the model
invented one, which is a failure rather than an argument to parse.

**2026-07-26** — M2 done. Operand verification is a rule built by
`operand_rule(nameable, resolver=...)`, with resolution split into
`resolvers.py` behind a protocol. The domain supplies how its records are named,
because only it knows.

Running it on the odontogram found that the lexical resolver could not resolve a
tooth at all: FDI is two digits and short tokens were dropped as noise, so a
dictation saying "el 36" named nothing and the rule allowed a procedure on any
tooth. That is the second requirement the second domain produced, and the
argument for having built one.

**2026-07-26** — M1 done. `domain.py` makes a domain a value rather than a
module of globals, and `odontogram.py` is the second one — enough to prove they
coexist and to force out what a single domain never needed.

Building it produced a requirement the design did not have: **chained
arguments**. It also exposed that `tool_schemas` computed each argument
independently, which for a chained method offers combinations that do not exist
and which no JSON Schema can rule out. A chained method is now offered as one
enumerated choice over real combinations, which is the design law applied where
it also happens to save two round trips.

The clinical manifest keeps its module-level names, delegating to the domain, so
the six callers written before domains existed did not have to change at once.

**2026-07-26** — M0 done. `firmware/` carries `Rule`, `Runtime`, `Proposal` and
a decision `Journal` separate from the store's effect journal — a blocked action
leaves no trace in the data, and that is exactly the trace an audit wants. The
three clinical rule families are wired with priorities so nobody is asked to
inspect something that was going to be blocked. 17 tests, no model required.

Also folded `BADMEMORY.md` into §3 as a corroboration section: it was written in
the mask's vocabulary and its one load-bearing claim — that independent work on
frontier models found the same shape, with the goal that "resembles a legitimate
user preference" at their highest success rate — belongs beside the rest of the
state of the art rather than in a file of its own.

**2026-07-26** — Created. Records the state of the art as surveyed, the adoption
of AgentSpec's rule model, the two additions and the measurements behind them,
and the block layout as planned for M0.
