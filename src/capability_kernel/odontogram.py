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

#: A stand-in for the payer's nomenclador. The real one is loaded per country
#: and per date; what matters structurally is that it is keyed by surface, so a
#: code valid on an occlusal surface is not offered on an incisal one.
CODES: dict[str, tuple[str, ...]] = {
    "occlusal": ("D2391", "D2392", "D2140"),
    "incisal":  ("D2330", "D2331"),
    "mesial":   ("D2331", "D2140"),
    "distal":   ("D2331", "D2140"),
    "buccal":   ("D2332",),
    "lingual":  ("D2332",),
}


@dataclass
class Tooth:
    fdi: str
    present: bool = True

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
        return CODES.get(surface, ())

    def record(self, tooth: str, surface: str, code: str) -> str:
        entry = f"{code} on {tooth} {surface}"
        self.journal.append(entry)
        return f"recorded {entry}"

    def get(self, fdi: str) -> Tooth | None:
        """Present so a domain-agnostic runtime can resolve a target."""
        return self.teeth.get(fdi)


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
ODONTOGRAM = Domain(
    name="odontogram",
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
