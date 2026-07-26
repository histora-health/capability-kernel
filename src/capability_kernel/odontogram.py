"""A second domain, to prove one process holds two — and to find what breaks.

Deliberately minimal; M3 grows it into the procedure-coding case. What it has to
carry now is the property the clinical domain does not: **chained arguments**.
Which surfaces exist depends on which tooth was chosen, and which codes are
valid depends on the surface. That dependency is what a JSON Schema cannot
state, and it is also what breaks a surface that computes each argument
independently.

The constraint here is *structural* rather than permission-based, which is the
reason this case exists. Tooth 11 has an incisal surface and no occlusal one —
that is not a prohibition the model is fighting, it is a category that does not
exist, and the clinician is not asking for it. Silent substitution should
largely not arise, and if it does anyway that is worth knowing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain import Domain, Method

#: FDI notation. Anterior teeth end in 1–3, posterior in 4–8.
ANTERIOR_SURFACES = ("mesial", "distal", "buccal", "lingual", "incisal")
POSTERIOR_SURFACES = ("mesial", "distal", "buccal", "lingual", "occlusal")

#: A value set: which procedure codes are billable on which surface.
#:
#: Keyed by surface because that is the structural fact — a code for a
#: multi-surface posterior restoration is not billable on an incisal edge — and
#: because keying it this way is what lets the surface narrow the codes.
#:
#: Real deployments load this per payer and per date. `ValueSet` exists so that
#: swapping the nomenclador is swapping a file rather than editing a manifest,
#: which is the whole claim about codes: a hallucinated one is a rejected claim,
#: and enumeration makes the rate zero by construction rather than by review.
DEFAULT_CODES: dict[str, tuple[str, ...]] = {
    "occlusal": ("D2391", "D2392", "D2140"),
    "incisal":  ("D2330", "D2331"),
    "mesial":   ("D2331", "D2140"),
    "distal":   ("D2331", "D2140"),
    "buccal":   ("D2332",),
    "lingual":  ("D2332",),
}

#: What each code means, for the proposal a person confirms. A person approving
#: `D2391` is approving a string; approving "one-surface posterior composite" is
#: approving a procedure.
CODE_NAMES: dict[str, str] = {
    "D2140": "amalgam, one surface",
    "D2330": "resin composite, one surface, anterior",
    "D2331": "resin composite, two surfaces, anterior",
    "D2332": "resin composite, three surfaces, anterior",
    "D2391": "resin composite, one surface, posterior",
    "D2392": "resin composite, two surfaces, posterior",
}


@dataclass
class ValueSet:
    """The codes in force for a payer, keyed by surface.

    A class rather than a dict so that loading one is an explicit act with a
    name attached to it. Which payer's codes were in force is the first thing
    asked when a claim is rejected.
    """

    payer: str
    by_surface: dict[str, tuple[str, ...]]

    def for_surface(self, surface: str) -> tuple[str, ...]:
        return self.by_surface.get(surface, ())

    def describe(self, code: str) -> str:
        return CODE_NAMES.get(code, code)

    @classmethod
    def default(cls) -> "ValueSet":
        return cls(payer="default", by_surface=dict(DEFAULT_CODES))


@dataclass
class Tooth:
    fdi: str
    present: bool = True

    @property
    def id(self) -> str:
        return self.fdi

    @property
    def name(self) -> str:
        """What a clinician would say. The resolver matches on this, so
        "el 36" and "tooth 36" both reach the same record."""
        kind = "incisor" if self.anterior else "molar"
        return f"tooth {self.fdi} {kind}"

    @property
    def anterior(self) -> bool:
        return self.fdi[-1] in "123"

    @property
    def surfaces(self) -> tuple[str, ...]:
        return ANTERIOR_SURFACES if self.anterior else POSTERIOR_SURFACES


@dataclass
class Odontogram:
    """One patient's dentition, and what has been recorded on it."""

    teeth: dict[str, Tooth] = field(default_factory=dict)
    journal: list[str] = field(default_factory=list)
    value_set: ValueSet = field(default_factory=ValueSet.default)

    def present_teeth(self) -> list[str]:
        """Teeth that can carry a procedure.

        A tooth recorded as absent is not a permission question — there is
        nothing there to treat, so it is not an option.
        """
        return [t.fdi for t in self.teeth.values() if t.present]

    def surfaces_of(self, fdi: str) -> tuple[str, ...]:
        tooth = self.teeth.get(fdi)
        return tooth.surfaces if tooth and tooth.present else ()

    def codes_for(self, surface: str) -> tuple[str, ...]:
        return self.value_set.for_surface(surface)

    def record(self, tooth: str, surface: str, code: str) -> str:
        entry = f"{code} on {tooth} {surface}"
        self.journal.append(entry)
        return f"recorded {self.value_set.describe(code)} on tooth {tooth}, {surface}"

    def decline(self, reason: str) -> str:
        return reason

    def describe(self) -> str:
        """The dentition, for the model's context."""
        present = ", ".join(self.present_teeth())
        missing = ", ".join(t.fdi for t in self.teeth.values() if not t.present)
        lines = [f"Teeth present: {present}"]
        if missing:
            lines.append(f"Missing: {missing}")
        if self.journal:
            lines.append("Recorded so far: " + "; ".join(self.journal))
        return "\n".join(lines)

    def get(self, fdi: str) -> Tooth | None:
        """Present so a domain-agnostic runtime can resolve a target."""
        return self.teeth.get(fdi)

    def nameable(self) -> list[Tooth]:
        """Every tooth, absent ones included.

        Absent teeth resolve even though nothing can be recorded on them —
        which is the same reason closed records resolve in the clinical
        domain. A dictation naming tooth 16 must be recognisable as being about
        tooth 16, or the operand rule cannot tell a substitution from a request
        it simply did not understand.
        """
        return list(self.teeth.values())


def demo_odontogram() -> Odontogram:
    """Upper right quadrant, with 16 missing.

    The missing tooth is the point: it is present in the notation and absent
    from every option, without a rule saying so.
    """
    o = Odontogram()
    for fdi in ("11", "12", "13", "14", "15", "16", "17", "18"):
        o.teeth[fdi] = Tooth(fdi, present=(fdi != "16"))
    return o


#: `surface` depends on `tooth`, and `code` on `surface`. Each source takes the
#: arguments chosen so far, which is what lets one model call produce a complete
#: and valid procedure instead of three round trips.
def odontogram_entities(store) -> list:
    from .resolvers import Entity
    return [Entity(t.id, t.name) for t in store.nameable()]


ODONTOGRAM = Domain(
    name="odontogram",
    nameable=odontogram_entities,
    methods={
        "record_procedure": Method(
            name="record_procedure",
            summary="Record a procedure on a tooth surface.",
            args={
                "tooth":   lambda s, chosen: s.present_teeth(),
                "surface": lambda s, chosen: s.surfaces_of(chosen.get("tooth", "")),
                "code":    lambda s, chosen: s.codes_for(chosen.get("surface", "")),
            },
        ),
        "decline": Method(
            name="decline",
            summary="Take no action, and say why.",
            args={"reason": None},
        ),
    },
    virtual=("decline",),
)
