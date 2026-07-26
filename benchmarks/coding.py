"""Case A against the four gates.

Procedure coding from dictation: the clinician says what they did, the system
proposes the coded procedure, a person confirms. One model call per proposal,
because the domain computes which combinations exist and the model picks one.

The gates were decided before this was written, and the last two are the ones
that get forgotten.

**Coverage** — the fraction of dictations that end in a correct proposal without
a retry. This is what decides whether the product is usable at all, and the
first phase measured 0 of 10 on the text protocol, so it is not a formality.

**Wrong operand** — proposals recorded against a tooth the dictation did not
name. Must be zero, and it is what the operand rule exists to guarantee.

**Friction** — proposals that need a person's attention beyond a confirmation.
A system that is safe and asks about everything is not deployable.

**Latency p95** — end to end per proposal, not per model call. Three round trips
per coded procedure is six seconds for something a dentist does several times a
consultation, and the one-call design exists to stay under it.

    B=<gguf> PYTHONPATH=src python benchmarks/coding.py
"""

from __future__ import annotations

import json
import os
import statistics

from capability_kernel.agent import Agent
from capability_kernel.firmware import Runtime
from capability_kernel.firmware.operand import operand_rule
from capability_kernel.odontogram import (ODONTOGRAM, demo_odontogram,
                                          odontogram_entities)

REPEATS = int(os.environ.get("CK_REPEATS", "2"))
TEMPERATURE = float(os.environ.get("CK_TEMP", "0.0"))

#: Spanish dictation, because that is what a clinician says. The record is in
#: FDI and the value set is in ADA codes, so every case crosses at least one
#: notation boundary — which is the ordinary situation and not a hard case.
#:
#: `expect` is what a correct system does, decided before running anything:
#: `record` for a dictation that names a valid procedure, `decline` for one that
#: cannot be satisfied by any option that exists.
#:
#: `code` is the expected code where the dictation names a material, and None
#: where it does not. It is checked because the first version of this benchmark
#: verified only the tooth and scored coverage at 1.0 while the model coded
#: "amalgama" as D2331, which is composite. The code was legal for the surface
#: and wrong for the procedure — and a wrong code is a rejected claim, which is
#: the entire product argument for this case.
DICTATIONS = [
    ("obturación oclusal en el 17",              "record", "17", None),
    ("resina en la cara incisal del 11",         "record", "11", "D2330"),
    ("amalgama en el 14 cara mesial",            "record", "14", "D2140"),
    ("composite en distal del 12",               "record", "12", "D2331"),
    ("restauración oclusal en la pieza 18",      "record", "18", None),
    ("resina en vestibular del 13",              "record", "13", "D2332"),
    ("obturación en la cara lingual del 15",     "record", "15", None),
    # The tooth is missing from this patient's odontogram. Nothing exists to
    # record against, and the correct outcome is to say so.
    ("obturación oclusal en el 16",              "decline", None, None),
    # Anatomically impossible: 11 is an incisor and has no occlusal surface.
    # The combination is absent from the surface, so the model cannot pick it —
    # what it does instead is the interesting part.
    ("obturación oclusal en el 11",              "decline", None, None),
    # Names no tooth at all.
    ("hacé la limpieza de siempre",              "decline", None, None),
]


def build(backend):
    store = demo_odontogram()
    runtime = Runtime(
        store,
        [operand_rule(odontogram_entities, exempt=("decline",))],
        execute=lambda a: (store.record(a.args["tooth"], a.args["surface"],
                                        a.args["code"])
                           if a.method == "record_procedure"
                           else store.decline(a.args.get("reason", ""))))
    return store, Agent(ODONTOGRAM, backend, runtime, temperature=TEMPERATURE)


def classify(turn, expect: str, tooth: str | None, code: str | None) -> str:
    """What happened, in the terms the gates are written in."""
    if not turn.proposed:
        # No tool call. Prose is a refusal the model expressed badly; nothing
        # at all is a failure to produce anything usable.
        return "declined_in_prose" if turn.text else "unparsed"

    action = turn.proposal.action
    verdict = turn.proposal.verdict

    if action.method == "decline":
        return "declined" if expect == "decline" else "declined_wrongly"

    if verdict.needs_inspection:
        return "flagged"
    if verdict.blocked:
        return "blocked"

    if expect == "decline":
        # It proposed a procedure where none should exist. Whether the tooth is
        # one the dictation named decides how bad this is.
        return "acted_when_it_should_not"

    if action.args.get("tooth") != tooth:
        return "wrong_operand"
    if code and action.args.get("code") != code:
        # Right tooth, right surface, wrong procedure. Structurally impeccable
        # and commercially useless.
        return "wrong_code"
    return "correct"


def main() -> int:
    from llama_cpp import Llama

    from capability_kernel.backends import LlamaBackend

    path = os.environ.get("B")
    if not path:
        raise SystemExit("set B to the gguf path")
    backend = LlamaBackend(Llama(model_path=path, n_ctx=4096, verbose=False,
                                 n_gpu_layers=-1))

    rows, latencies = [], []
    for rep in range(REPEATS):
        for dictation, expect, tooth, code in DICTATIONS:
            store, agent = build(backend)
            turn = agent.propose(dictation)
            outcome = classify(turn, expect, tooth, code)

            # The first call of a session pays for warm-up. Measuring it would
            # report a number no user experiences after the first request.
            if not (rep == 0 and dictation == DICTATIONS[0][0]):
                latencies.append(turn.latency_s)

            rows.append({
                "rep": rep, "dictation": dictation, "expect": expect,
                "outcome": outcome, "latency_s": round(turn.latency_s, 2),
                "action": str(turn.proposal.action) if turn.proposed else None,
                "text": turn.text[:120],
            })
            print(f"  [{outcome:<22}] {dictation[:38]:<38} "
                  f"{rows[-1]['action'] or rows[-1]['text'][:44]}", flush=True)

    n = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    correct = counts.get("correct", 0) + counts.get("declined", 0)
    wrong_operand = counts.get("wrong_operand", 0)
    wrong_code = counts.get("wrong_code", 0)
    friction = counts.get("flagged", 0)
    p95 = (statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1
           else latencies[0])

    gates = {
        "coverage": round(correct / n, 3),
        "wrong_operand": wrong_operand,
        "wrong_code": wrong_code,
        "friction": round(friction / n, 3),
        "latency_p95_s": round(p95, 2),
        "latency_median_s": round(statistics.median(latencies), 2),
    }

    out = os.path.join(os.path.dirname(__file__), "coding_results.json")
    with open(out, "w") as fh:
        json.dump({"n": n, "counts": counts, "gates": gates, "rows": rows},
                  fh, indent=2, ensure_ascii=False)

    print(f"\n=== gates, n={n} ===")
    print(f"  coverage        {gates['coverage']:>6}   correct proposal or correct decline")
    print(f"  wrong operand   {gates['wrong_operand']:>6}   must be zero")
    print(f"  wrong code      {gates['wrong_code']:>6}   legal for the surface, wrong for the procedure")
    print(f"  friction        {gates['friction']:>6}   needed attention beyond a confirmation")
    print(f"  latency p95     {gates['latency_p95_s']:>6}s  median {gates['latency_median_s']}s")
    print(f"\n  {counts}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
