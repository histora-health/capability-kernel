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
    assert not any("tooth=16" in c for c in ODONTOGRAM.action_strings(odo, "record_procedure"))


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


# ── The operand rule, on a domain it was not written for ─────────────────────


def test_the_same_rule_works_on_a_second_domain(odo):
    """The reason the resolver takes (id, name) pairs rather than store objects.

    Nothing in `operand_rule` knows about teeth, and nothing in `Odontogram`
    knows about resolvers.
    """
    from capability_kernel.firmware import Action, Context, Runtime
    from capability_kernel.firmware.operand import operand_rule
    from capability_kernel.odontogram import odontogram_entities

    runtime = Runtime(odo, [operand_rule(odontogram_entities, exempt=("decline",))],
                      execute=lambda a: "recorded")
    dictation = "obturación oclusal en el 17"

    right = runtime.evaluate(
        Action("record_procedure", {"target": "17", "tooth": "17",
                                    "surface": "occlusal", "code": "D2391"}),
        Context(request=dictation))
    wrong = runtime.evaluate(
        Action("record_procedure", {"target": "15", "tooth": "15",
                                    "surface": "occlusal", "code": "D2391"}),
        Context(request=dictation))

    assert right.allowed
    assert wrong.needs_inspection


def test_a_two_digit_tooth_number_carries_reference(odo):
    """The length filter made every tooth invisible.

    FDI names a tooth in two digits, and dropping short tokens as noise dropped
    the only word in the dictation that identified the record. The rule then
    allowed a procedure on any tooth at all — found by running it on this
    domain, which is what this domain is for.
    """
    from capability_kernel.odontogram import odontogram_entities
    from capability_kernel.resolvers import DEFAULT

    named = DEFAULT.candidates(odontogram_entities(odo), "obturación en el 17")
    assert [r.entity.id for r in named] == ["17"]


def test_an_absent_tooth_still_resolves(odo):
    """It cannot be acted on and must still be nameable.

    Same reason closed records resolve in the clinical domain: a dictation
    about tooth 16 has to be recognisable as being about tooth 16, or the rule
    cannot tell a substitution from a request it did not understand.
    """
    from capability_kernel.odontogram import odontogram_entities
    from capability_kernel.resolvers import DEFAULT

    named = DEFAULT.candidates(odontogram_entities(odo), "algo en el 16")
    assert [r.entity.id for r in named] == ["16"]
    assert "16" not in ODONTOGRAM.legal_values(odo, "record_procedure", "tooth")


def test_the_resolver_is_swappable_without_touching_the_rule(odo):
    """The split exists because the rule is settled and the resolver is not."""
    from capability_kernel.firmware import Action, Context, Runtime
    from capability_kernel.firmware.operand import operand_rule
    from capability_kernel.odontogram import odontogram_entities

    class NeverResolves:
        def candidates(self, entities, message):
            return []

    runtime = Runtime(odo, [operand_rule(odontogram_entities,
                                         resolver=NeverResolves(),
                                         exempt=("decline",))],
                      execute=lambda a: "recorded")
    # A resolver that names nothing cannot object to anything, which is the
    # correct behaviour rather than a degenerate one.
    assert runtime.evaluate(
        Action("record_procedure", {"target": "15"}),
        Context(request="obturación oclusal en el 17")).allowed
