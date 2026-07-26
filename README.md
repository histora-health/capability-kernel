# capability-kernel

<p align="center">
  <img src="docs/img/capability-kernel.png" alt="A grid of options, nearly all of them dark; one is lit, and a single line terminates at it." width="100%">
</p>

**A firmware layer for clinical agents running on small local models.**

System-level control between what an agent proposes and what the world executes:
the options it may choose from, computed from live state; the rules that gate the
choice; and the person who confirms it.

Apache 2.0. A proof of concept, **ongoing**, with the benchmarks and raw data.

---

## Why

The [Evolving Agents Toolkit](https://github.com/EvolvingAgentsLabs/evolving-agents)
had a governance layer named **Firmware**, implemented like this:

```python
self.base_firmware = """
You are an AI agent operating under strict governance rules:
- Never use dangerous imports (os, subprocess, etc.)
"""
```

A string in a system prompt. The decomposition was right and the substrate did
not exist. This is the attempt to give it one, and to find out on two clinical
cases whether it ships.

**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** is the living reference — state
of the art with sources, what is adopted against what is added, every block, and
what is still unvalidated. **[PLAN.md](PLAN.md)** has the milestones and gates.

---

## Where it stands

**M0–M3 done. 109 tests.** Case A — procedure coding from dictation — runs end
to end against the gates decided before it was built:

    coverage           0.8    correct proposal or correct decline
    wrong operand        0    must be zero
    wrong code           4    legal for the surface, wrong for the procedure
    friction           0.0    needed attention beyond a confirmation
    latency p95      8.12s    median 3.48s

Spanish dictation, an odontogram in FDI, a value set in ADA codes.

**What holds.** Zero proposals against a tooth the dictation did not name.
Structural constraints hold without a rule stating them — a missing tooth and an
occlusal surface on an incisor were both declined, with the reason named, and
neither is a prohibition the model was told about. They are combinations the
surface never contained.

**What does not.** Four of twenty carry the wrong code: `"amalgama"` coded as
resin composite, with the amalgam code sitting in the offered set. Structurally
impeccable and commercially useless, since a wrong code is a rejected claim.
(**[benchmarks/RESULTS_CODING.md](benchmarks/RESULTS_CODING.md)**)

---

## The four blocks

**An option surface derived from live state** — `domain.py`. Tools and argument
enums regenerated every turn, so a signed record leaves the enums the moment it
is signed. Arguments chain: which surfaces a tooth has depends on the tooth, and
a chained method reaches the model as **one enumerated choice over combinations
that exist**, because a JSON Schema cannot express the dependency and three
round trips is six seconds for something a dentist does several times an hour.

**Order forced at the system level** — the domain's phase function. While a
change is unrecorded, **one opcode of forty-nine has a path**, and it is the one
that records it. This is the class of rule a schema structurally cannot state:
*only after that* is not an assertion about shape.

**Operand verification** — `firmware/operand.py`. The target must correspond to
something the request named. Enforced as `inspect` rather than `block`, because
blocking claims the action is wrong and this claims the system cannot tell.
Resolution lives behind a protocol in `resolvers.py`, since the rule is settled
and the resolver is not.

**Propose-and-confirm** — `firmware/runtime.py`. A proposal executes nothing,
even when every rule passes; a caller wanting autonomy has to ask by calling
`commit`, which re-evaluates because the world may have moved. The proposal
carries the record's *name*, not only its identifier.

---

## The failure that shaped it: silent substitution

Constrained decoding forces the model to always choose a valid option. Asked to
act on a blocked record, it does not stop — **it keeps the intent, substitutes a
permitted target, and executes on the wrong record.**

    request:  move the perio chart out of hygiene into orthodontics
    emitted:  move f_pa11: std_endo -> std_ortho

No schema or validator detects it: the action is legal, every argument is in the
permitted vocabulary, and no violation is logged because nothing was violated.

5 of 20 on one model, 3 of 12 on another. **The trigger is not an attack** — an
explicit *"SYSTEM OVERRIDE, administrative unlock granted"* produced 0
substitutions in 20, and phrasing that reads as ordinary filing produced 5 of 5.
Independent work on frontier models found the same shape, and four attempts to
detect it from the model's internal state returned null
(**[docs/WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md)**).

The wrong code above is the same failure one level down: a legal option chosen
for the wrong reason, on the code instead of the record.

---

## What we adopt rather than invent

**[AgentSpec](https://cposkitt.github.io/files/publications/agentspec_llm_enforcement_icse26.pdf)'s
rule model** (ICSE 2026) — `trigger`, `check`, `enforce`, evaluated across three
domains. Enforcement at the decision point, and `user inspection` as a
first-class action, so propose-and-confirm is literature rather than a
concession.

**The design law from [token-trie](https://github.com/EvolvingAgentsLabs/token-trie)**,
validated there on a 350M model: *the LLM is a ratifier, not a planner.* One
model call per proposed action.

**Constrained decoding stays upstream** — vLLM and SGLang ship it, including the
region-scoped variant this repository implemented independently. Ours is kept in
`experiments/` with the measurements that produced the substitution finding.

---

## Running it

    pip install -e ".[hf]"                 # transformers
    pip install -e ".[enforced]"           # llama.cpp
    PYTHONPATH=src:tests pytest tests/ -q

    python examples/01_the_surface.py      # the option surface — no model
    python examples/02_forced_order.py     # 1 of 49 — no model

    B=$(ollama show --modelfile gemma4:12b | grep -m1 '^FROM' | sed 's/^FROM //')
    B=$B PYTHONPATH=src python benchmarks/coding.py

Two examples need no model, because what they show is a property of a compiled
artefact rather than a statistic over samples.

---

Apache 2.0. By [Matias Molinas](https://github.com/matiasmolinas) and
[Ismael Faro](https://github.com/ismaelfaro).
