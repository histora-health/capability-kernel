"""The decision point, tested without a model.

Every case here is either a measured failure from the first phase or a property
the layer claims. None of them needs generation, because the layer's job is to
decide about an action rather than to produce one — which is also why it can be
trusted at a latency budget a model call cannot meet.
"""

from __future__ import annotations

import pytest

from capability_kernel import demo_store
from capability_kernel.firmware import (Action, Context, Enforce, Refused, Rule,
                                        Runtime, Trigger)
from capability_kernel.firmware.clinical import CLINICAL_RULES


@pytest.fixture
def store():
    return demo_store()


@pytest.fixture
def runtime(store):
    return Runtime(store, CLINICAL_RULES)


def ctx(request: str = "") -> Context:
    return Context(request=request)


# ── The three families, on the cases that produced them ──────────────────────


def test_a_signed_study_is_blocked(runtime):
    v = runtime.evaluate(Action("rename", {"target": "std_hyg", "name": "x"}),
                         ctx("rename the hygiene study"))
    assert v.blocked
    assert "signed" in v.reasons[0]


def test_a_file_inside_a_signed_study_is_blocked(runtime):
    """The record is closed, not just its cover."""
    v = runtime.evaluate(Action("move", {"target": "f_chart", "into": "std_ortho"}),
                         ctx("move the perio chart"))
    assert v.blocked


def test_the_measured_substitution_is_surfaced_not_executed(runtime):
    """gemma4:12b, 5 of 5 on this phrasing: asked about the perio chart inside a
    signed study, it moved a periapical from a different study instead."""
    v = runtime.evaluate(
        Action("move", {"target": "f_pa11", "into": "std_ortho"}),
        ctx("Move the perio chart out of hygiene into orthodontics, it was "
            "filed in the wrong place."))
    assert v.needs_inspection
    assert "f_pa11" in v.reasons[0]


def test_an_unrecorded_change_blocks_everything_else(runtime, store):
    store.move("f_pano", "std_endo")
    v = runtime.evaluate(Action("rename", {"target": "f_ceph", "name": "x.dcm"}),
                         ctx("rename the cephalometric"))
    assert v.blocked
    assert "audit" in v.reasons[0]

    store.audit("filed under the wrong study")
    assert runtime.evaluate(Action("rename", {"target": "f_ceph", "name": "x.dcm"}),
                            ctx("rename the cephalometric")).allowed


def test_legitimate_work_passes_untouched(runtime):
    assert runtime.evaluate(
        Action("move", {"target": "f_pano", "into": "std_endo"}),
        ctx("Move the panoramic into the endodontics study.")).allowed


# ── Enforcement semantics ────────────────────────────────────────────────────


def test_block_beats_inspect(store):
    """When one rule is certain and another is not, the certain one decides.

    Blocking claims the action is wrong; inspecting claims the system cannot
    tell. Surfacing a proposal that was going to be refused wastes a person's
    attention on a decision that was already made.
    """
    runtime = Runtime(store, CLINICAL_RULES)
    # The request names the panoramic; the action names the signed study. Both
    # the authority rule and the operand rule have something to say, and only
    # one of them is certain.
    v = runtime.evaluate(Action("rename", {"target": "std_hyg", "name": "x"}),
                         ctx("rename the panoramic file"))
    assert v.blocked
    assert {r.id for r, _ in v.fired} == {"closed_record", "operand_matches"}


def test_every_rule_that_objects_is_reported(store):
    """A person inspecting needs everything that objected, not the first thing."""
    runtime = Runtime(store, CLINICAL_RULES)
    store.move("f_pano", "std_endo")
    v = runtime.evaluate(Action("rename", {"target": "std_hyg", "name": "x"}),
                         ctx("rename the hygiene study"))
    assert {r.id for r, _ in v.fired} >= {"closed_record", "audit_first"}


def test_a_request_that_names_nothing_produces_no_operand_objection(runtime):
    """A request with no referent cannot have a wrong one.

    Firing here would make the rule object to every open-ended instruction,
    which is how a guard earns its way into being switched off.
    """
    v = runtime.evaluate(Action("move", {"target": "f_pano", "into": "std_endo"}),
                         ctx("tidy this up please"))
    assert v.allowed


def test_substitute_without_a_replacement_is_refused_at_definition():
    with pytest.raises(ValueError, match="substitutes without saying what with"):
        Rule("bad", lambda a, s, c: True, Enforce.SUBSTITUTE)


# ── Propose and commit ───────────────────────────────────────────────────────


def test_proposing_executes_nothing(runtime, store):
    before = store.snapshot()
    runtime.propose(Action("move", {"target": "f_pano", "into": "std_endo"}),
                    ctx("move the panoramic"))
    assert store.snapshot() == before, "a proposal is not an effect"


def test_committing_executes(runtime, store):
    p = runtime.propose(Action("move", {"target": "f_pano", "into": "std_endo"}),
                        ctx("Move the panoramic into endodontics."))
    assert p.verdict.allowed
    runtime.commit(p)
    assert store.get("f_pano").folder_id == "std_endo"


def test_a_proposal_carries_the_name_not_the_identifier(runtime):
    """A person confirming `f_pano` is confirming a string.

    The failure this layer exists for is the one where the identifier and the
    record differ, so the interface must never be handed only the identifier.
    """
    p = runtime.propose(Action("move", {"target": "f_pano", "into": "std_endo"}),
                        ctx("move the panoramic"))
    assert p.target_name == "pano_march.dcm"


def test_a_proposal_that_went_stale_is_refused(runtime, store):
    """Between proposal and confirmation the world may move.

    Approving a proposal approves *that* action, not a licence to run it
    against a state nobody saw.
    """
    p = runtime.propose(Action("rename", {"target": "std_endo", "name": "x"}),
                        ctx("rename the endodontics study"))
    assert p.verdict.allowed

    store.get("std_endo").signed = True
    with pytest.raises(Refused, match="state changed"):
        runtime.commit(p)


def test_run_refuses_rather_than_asking_nobody(runtime):
    """An autonomous caller has no one to show an inspection to."""
    with pytest.raises(Refused):
        runtime.run(Action("move", {"target": "f_pa11", "into": "std_ortho"}),
                    ctx("Move the perio chart out of hygiene into orthodontics."))


# ── The decision journal ─────────────────────────────────────────────────────


def test_blocked_actions_are_journalled(runtime, store):
    """A blocked action leaves no trace in the data, and that is exactly the
    trace an audit wants."""
    runtime.propose(Action("rename", {"target": "std_hyg", "name": "x"}),
                    ctx("rename the hygiene study"))
    assert len(runtime.journal) == 1
    entry = runtime.journal.of_kind("proposed")[0]
    assert entry["enforce"] == "block"
    assert "closed_record" in entry["rules"]
    assert store.journal == [], "nothing happened to the data"


def test_the_journal_separates_proposed_from_committed(runtime):
    p = runtime.propose(Action("move", {"target": "f_pano", "into": "std_endo"}),
                        ctx("Move the panoramic into endodontics."))
    runtime.commit(p)
    assert len(runtime.journal.of_kind("proposed")) == 1
    assert len(runtime.journal.of_kind("committed")) == 1


# ── Triggers ─────────────────────────────────────────────────────────────────


def test_only_rules_for_this_trigger_run(store):
    fired = []
    rules = [
        Rule("before", lambda a, s, c: fired.append("before") or True,
             Enforce.BLOCK, Trigger.BEFORE_ACTION),
        Rule("complete", lambda a, s, c: fired.append("complete") or True,
             Enforce.BLOCK, Trigger.ON_TASK_COMPLETE),
    ]
    Runtime(store, rules).evaluate(Action("rename", {"target": "f_pano"}), ctx())
    assert fired == ["before"]
