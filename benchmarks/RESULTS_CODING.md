# Case A — procedure coding, against the four gates

Dictation in Spanish, an odontogram in FDI, a value set in ADA codes. One model
call per proposal: the domain computes which combinations exist and the model
picks one. gemma4:12b through llama.cpp, twenty proposals.

Reproduce: `B=<gguf> PYTHONPATH=src python benchmarks/coding.py`

## The gates

    coverage           0.8    correct proposal or correct decline
    wrong operand        0    must be zero
    wrong code           4    legal for the surface, wrong for the procedure
    friction           0.0    needed attention beyond a confirmation
    latency p95      8.12s    median 3.48s

## What works

**The operand gate holds.** Zero proposals against a tooth the dictation did not
name, across twenty. The dictation says "el 17" and the record says `17`, in a
different language from the value set, and the resolver connects them.

**Structural constraints are respected without a rule saying so.** Tooth 16 is
missing from this patient, so nothing can be recorded against it — and the model
declined, naming the reason:

> *The tooth 16 is listed as missing in the current record, and there is no
> option available to record a procedure on a missing tooth.*

Same for the anatomically impossible: 11 is an incisor, an occlusal surface does
not exist on it, and the model said so rather than picking a surface that does.
Neither of those is a prohibition it was told about. They are combinations the
surface never contained.

**And the vague dictation was declined** rather than guessed at, which is the
behaviour that makes the other nineteen trustworthy.

## What does not

**Four of twenty carry the wrong code.** Right tooth, right surface, wrong
procedure — the clearest being:

    "amalgama en el 14 cara mesial"  →  code=D2331   (resin composite)
                                        D2140 is amalgam, and was available

Structurally impeccable and commercially useless. A wrong code is a rejected
claim, which is the entire product argument for this case.

This is not a failure of enumeration: the correct code was in the offered set
and the model chose another one. It is the same class as the substitution
failure — a legal option chosen for the wrong reason — one level down, on the
code instead of the record.

**The first version of this benchmark did not catch it.** It verified the tooth
and reported coverage at 1.0. Checking only the operand measures the guard that
was built and not the product that was promised.

## What to do about it

The material is in the dictation and absent from the value set's structure: the
codes are keyed by surface, so both an amalgam and a composite code are offered
on a mesial surface and nothing distinguishes them at the point of choice.

Two options, and the first is the one the design already argues for. **Key the
value set by material as well as surface**, so "amalgama" narrows the set to one
code and the model chooses nothing. That is the program computing what it can
compute. **Or leave the choice with the model and show the code's meaning in
the proposal** — `CODE_NAMES` exists for this — so the person confirming sees
"resin composite" against a dictation that said amalgam.

The second is weaker and cheaper, and it is what propose-and-confirm is for. The
first removes the failure.

## Latency

Median 3.48s per proposal, p95 8.12s. The tail is the declines: the model writes
a paragraph explaining itself, and generation dominates. Capping the reason
would move p95 without touching the median, and the reason is what a person
reads, so it is not obviously worth it.

Against the ~2s per turn measured with an empty tool schema, the cost of
carrying sixty enumerated combinations in the prompt is about 1.5s.
