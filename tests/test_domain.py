"""Domains, chained arguments, and the ceiling.

The clinical domain's behaviour is covered by `test_store_and_manifest.py`,
which exercises it through the module-level functions. This covers what the
second domain forced into existence: arguments that depend on each other, and
what happens when they are offered to a model.
"""

from __future__ import annotations

import pytest

from capability_kernel import demo_store
from capability_kernel.domain import Domain, Method
from capability_kernel.manifest import CLINICAL
from capability_kernel.odontogram import ODONTOGRAM, Odontogram, Tooth, demo_odontogram


@pytest.fixture
def odo():
    return demo_odontogram()


# ── Two domains, one process ─────────────────────────────────────────────────


def test_both_domains_compute_their_own_surface(odo):
    """The reason `Domain` is a value rather than a module of globals."""
    assert CLINICAL.name != ODONTOGRAM.name
    assert set(CLINICAL.methods) != set(ODONTOGRAM.methods)
    assert CLINICAL.surface_size(demo_store())
    assert ODONTOGRAM.surface_size(odo)


def test_a_domain_does_not_assume_the_clinical_store(odo):
    """`Odontogram` shares no base class or field names with `ClinicalStore`."""
    assert ODONTOGRAM.legal_values(odo, "record_procedure", "tooth")


def test_virtual_methods_must_exist():
    with pytest.raises(ValueError, match="are not methods"):
        Domain(name="broken",
               methods={"act": Method("act", "", {})},
               virtual=("decline",))


# ── Chained arguments ────────────────────────────────────────────────────────


def test_surfaces_depend_on_the_tooth(odo):
    """An incisor has an incisal surface; a molar has an occlusal one.

    Computing this argument independently of the tooth is what would offer an
    occlusal surface on an incisor — a combination that does not exist, and
    which no JSON Schema can rule out.
    """
    anterior = ODONTOGRAM.legal_values(odo, "record_procedure", "surface", {"tooth": "11"})
    posterior = ODONTOGRAM.legal_values(odo, "record_procedure", "surface", {"tooth": "17"})

    assert "incisal" in anterior and "occlusal" not in anterior
    assert "occlusal" in posterior and "incisal" not in posterior


def test_codes_depend_on_the_surface(odo):
    incisal = ODONTOGRAM.legal_values(odo, "record_procedure", "code", {"surface": "incisal"})
    occlusal = ODONTOGRAM.legal_values(odo, "record_procedure", "code", {"surface": "occlusal"})
    assert incisal != occlusal


def test_an_absent_tooth_is_not_an_option(odo):
    """Not a permission question — there is nothing there to treat."""
    assert "16" not in ODONTOGRAM.legal_values(odo, "record_procedure", "tooth")
    assert not any("tooth=16" in c for c in ODONTOGRAM.opcode_strings(odo, "record_procedure"))


def test_chaining_is_detected(odo):
    assert ODONTOGRAM.is_chained("record_procedure")
    assert not ODONTOGRAM.is_chained("decline")
    assert not CLINICAL.is_chained("move")


def test_combinations_are_a_tree_not_a_product(odo):
    """A product of independent enums would include impossible rows."""
    combos = ODONTOGRAM.combinations(odo, "record_procedure")
    assert not any(c["tooth"][-1] in "123" and c["surface"] == "occlusal" for c in combos)
    assert all(c["code"] in ODONTOGRAM.methods["record_procedure"].args["code"](odo, c)
               for c in combos)


# ── How a chained method reaches the model ───────────────────────────────────


def test_a_chained_method_is_offered_as_one_choice(odo):
    """One call, not one per argument.

    Three round trips per coded procedure is six seconds at the measured ~2s
    per turn, for something a dentist does several times per consultation. The
    program already knows which combinations exist, so it presents them.
    """
    schema = next(t for t in ODONTOGRAM.tool_schemas(odo)
                  if t["function"]["name"] == "record_procedure")
    props = schema["function"]["parameters"]["properties"]

    assert list(props) == ["choice"], "one argument, not three"
    assert len(props["choice"]["enum"]) == len(ODONTOGRAM.combinations(odo, "record_procedure"))


def test_the_offered_choices_contain_no_impossible_combination(odo):
    schema = next(t for t in ODONTOGRAM.tool_schemas(odo)
                  if t["function"]["name"] == "record_procedure")
    enum = schema["function"]["parameters"]["properties"]["choice"]["enum"]

    assert not any("tooth=11" in e and "occlusal" in e for e in enum)
    assert not any("tooth=16" in e for e in enum)


def test_a_choice_round_trips_to_arguments(odo):
    schema = next(t for t in ODONTOGRAM.tool_schemas(odo)
                  if t["function"]["name"] == "record_procedure")
    choice = schema["function"]["parameters"]["properties"]["choice"]["enum"][0]

    args = ODONTOGRAM.parse_choice("record_procedure", choice)
    assert set(args) == {"tooth", "surface", "code"}
    assert args in ODONTOGRAM.combinations(odo, "record_procedure")


def test_an_unchained_method_keeps_its_arguments(odo):
    """Only chaining forces the collapse; the clinical domain is unaffected."""
    schema = next(t for t in CLINICAL.tool_schemas(demo_store())
                  if t["function"]["name"] == "move")
    assert set(schema["function"]["parameters"]["properties"]) == {"target", "into"}


def test_the_ceiling_refuses_loudly():
    """Enumeration has a limit, and an unreadable schema is a worse failure
    than a refused one."""
    many = Odontogram(teeth={str(i): Tooth(str(i)) for i in range(30, 130)})
    with pytest.raises(ValueError, match="over the ceiling"):
        ODONTOGRAM.tool_schemas(many)
