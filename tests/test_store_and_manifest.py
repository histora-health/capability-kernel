"""The capability surface is a projection of the world, not a fixed grammar.

These are the tests that matter for the thesis: the surface must *change* when
the store changes, and a signed study must be absent from it rather than
rejected by it. No model, no network.
"""

from __future__ import annotations

import pytest

from capability_kernel import ClinicalStore, StoreError, demo_store
from capability_kernel.manifest import legal_values, opcode_strings, surface_size, tool_schemas


@pytest.fixture
def store():
    return demo_store()


def enum_for(store, method, arg):
    props = next(t for t in tool_schemas(store) if t["function"]["name"] == method)
    return props["function"]["parameters"]["properties"][arg].get("enum")


# ── The surface follows the world ────────────────────────────────────────────


def test_renaming_removes_the_old_name_from_nothing_but_changes_the_world(store):
    """Ids are stable across renames; it is the *entity set* that gates them."""
    before = set(enum_for(store, "rename", "target"))
    store.rename("f_pano", "panoramic_march.dcm")
    after = set(enum_for(store, "rename", "target"))
    assert before == after, "renaming does not change which entities exist"
    assert store.get("f_pano").name == "panoramic_march.dcm"


def test_adding_a_file_widens_the_surface(store):
    before = surface_size(store)
    store.add_file("f_new", "extra.dcm", "std_endo")
    after = surface_size(store)

    assert after["rename"] == before["rename"] + 1
    assert after["move"] > before["move"]
    assert "f_new" in enum_for(store, "move", "target")


def test_a_file_that_does_not_exist_is_not_in_the_surface(store):
    """The core claim, in its weakest testable form: absence, not rejection."""
    assert "f_ghost" not in enum_for(store, "rename", "target")
    assert "f_ghost" not in enum_for(store, "move", "target")
    assert "f_ghost" not in (enum_for(store, "set_metadata", "target") or [])


# ── Signed studies are absent, not refused ───────────────────────────────────


def test_a_signed_study_is_absent_from_every_enum(store):
    assert "std_hyg" not in enum_for(store, "rename", "target")
    assert "std_hyg" not in enum_for(store, "move", "into")
    assert "std_hyg" not in enum_for(store, "set_metadata", "target")


def test_a_file_inside_a_signed_study_is_absent_too(store):
    """The record is closed, not just its cover."""
    for arg_method in (("rename", "target"), ("move", "target"), ("set_metadata", "target")):
        assert "f_chart" not in enum_for(store, *arg_method)


def test_signing_a_study_narrows_the_surface(store):
    assert "std_endo" in enum_for(store, "rename", "target")
    store.get("std_endo").signed = True
    assert "std_endo" not in enum_for(store, "rename", "target")
    assert "f_pa11" not in enum_for(store, "rename", "target"), "its files close with it"


def test_the_store_also_refuses_directly(store):
    """Belt and braces: even if a call reaches the store, it is refused.

    In the enforced arm this path is unreachable. In the baseline arm it is the
    only thing standing between a generated call and the record.
    """
    with pytest.raises(StoreError, match="signed"):
        store.rename("std_hyg", "Hygiene renamed")
    with pytest.raises(StoreError, match="signed"):
        store.move("f_chart", "std_endo")


# ── Operations ───────────────────────────────────────────────────────────────


def test_move_puts_the_file_in_the_new_study(store):
    store.move("f_pano", "std_endo")
    assert {f.id for f in store.files_in("std_endo")} == {"f_pa11", "f_pano"}
    assert "f_pano" not in {f.id for f in store.files_in("std_ortho")}


def test_name_collisions_are_refused_within_a_folder_only(store):
    store.add_file("f_dup", "pano_march.dcm", "std_endo")
    with pytest.raises(StoreError, match="already in"):
        store.move("f_dup", "std_ortho")
    # The same name in a different folder is fine.
    assert store.get("f_dup").folder_id == "std_endo"


def test_metadata_keys_and_values_are_controlled(store):
    store.set_metadata("f_pano", "modality", "panoramic")
    assert store.get("f_pano").metadata["modality"] == "panoramic"

    with pytest.raises(StoreError, match="unknown metadata key"):
        store.set_metadata("f_pano", "diagnosis", "caries")
    with pytest.raises(StoreError, match="not a legal value"):
        store.set_metadata("f_pano", "modality", "mri")


def test_free_text_keys_take_free_text(store):
    store.set_metadata("f_pano", "note", "patient tolerated well")
    assert legal_values(store, "set_metadata", "value") is None


def test_invalid_names_are_refused(store):
    for bad in ("", " leading", "sla/sh", "x" * 80):
        with pytest.raises(StoreError):
            store.rename("f_pano", bad)


# ── There is no delete ───────────────────────────────────────────────────────


def test_delete_is_not_in_the_manifest():
    """Deliberate. The clearest proof of the mechanism is a capability that
    structurally does not exist, and it makes the adversarial test unambiguous."""
    from capability_kernel.manifest import MANIFEST

    assert "delete" not in MANIFEST
    assert not any(t["function"]["name"] == "delete" for t in tool_schemas(demo_store()))
    assert not hasattr(ClinicalStore, "delete")


# ── The trie's input ─────────────────────────────────────────────────────────


def test_opcode_strings_enumerate_real_ids(store):
    ops = opcode_strings(store, "move")
    assert "move(target=f_pano, into=std_endo)" in ops
    assert not any("std_hyg" in o for o in ops), "signed study is not a destination"
    assert len(ops) == surface_size(store)["move"]


def test_free_text_arguments_become_slots(store):
    ops = opcode_strings(store, "rename")
    assert all("name={slot}" in o for o in ops)


def test_the_surface_is_small_enough_to_enumerate(store):
    """Enumeration has a ceiling. Watching the number is how you find out
    before a demo does."""
    assert max(surface_size(store).values()) < 512


# ── Telling a real action from a legal no-op ─────────────────────────────────


def test_a_snapshot_changes_only_when_something_changed(store):
    """The bound on the loop the mask cannot prevent.

    Renaming a file to the name it already has is a fully legal opcode. The
    surface has no reason to exclude it, so the chat loop has to notice it
    another way.
    """
    before = store.snapshot()
    store.rename("f_pano", store.get("f_pano").name)
    assert store.snapshot() == before, "a rename to the same name changed nothing"

    store.rename("f_pano", "different.dcm")
    assert store.snapshot() != before


def test_the_snapshot_sees_every_kind_of_change(store):
    for act in (lambda: store.rename("f_pano", "x.dcm"),
                lambda: store.move("f_pano", "std_endo"),
                lambda: store.set_metadata("f_pano", "stage", "pre-op")):
        before = store.snapshot()
        act()
        assert store.snapshot() != before
