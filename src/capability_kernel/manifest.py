"""The capability manifest: what exists, and what may be done right now.

One declaration, two compilation targets.

* :func:`tool_schemas` — JSON-schema tool definitions for a model with native
  function calling. The model is *asked* to stay inside them.
* the token trie (``compiler.py``) — the same surface as a logit mask. The model
  *cannot* leave them.

Both read the same manifest and the same store, which is the point: the two arms
differ in enforcement, not in what they consider legal. Anything else would make
the comparison meaningless.

Argument values are **enumerated from live state**, not validated against it. The
schema for ``rename`` does not say "a string"; it says which entity ids exist and
are renameable at this moment. A signed study is not in the enum, so in the
enforced arm it is unnameable — and in the baseline arm the model was told, and
may still say it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .store import METADATA_KEYS, METADATA_VALUES, ClinicalStore

#: There is no ``delete``. Deliberately: the clearest proof of the mechanism is
#: a capability that structurally does not exist, and it makes the adversarial
#: test unambiguous. Do not add one to "make the demo more useful".
METHODS = ("rename", "move", "set_metadata")


@dataclass(frozen=True)
class Method:
    name: str
    summary: str
    #: arg name -> how its legal values are produced.
    #: A callable takes the store and returns the enumerated values; a tuple is
    #: a fixed vocabulary; ``None`` means free text (a slot, in trie terms).
    args: dict


def _ids(entities) -> list[str]:
    return [e.id for e in entities]


MANIFEST: dict[str, Method] = {
    "rename": Method(
        name="rename",
        summary="Rename a study or a file. Signed studies cannot be renamed.",
        args={
            "target": lambda s: _ids(s.renameable()),
            "name": None,
        },
    ),
    "move": Method(
        name="move",
        summary="Move a file into another study.",
        args={
            "target": lambda s: _ids(s.movable()),
            "into": lambda s: _ids(s.move_targets()),
        },
    ),
    "set_metadata": Method(
        name="set_metadata",
        summary="Set one metadata field on a study or a file.",
        args={
            "target": lambda s: _ids(s.annotatable()),
            "key": tuple(METADATA_KEYS),
            "value": None,
        },
    ),
}


def legal_values(store: ClinicalStore, method: str, arg: str) -> list[str] | None:
    """The values ``arg`` may take right now, or ``None`` if it is free text."""
    spec = MANIFEST[method].args[arg]
    if spec is None:
        return None
    if callable(spec):
        return list(spec(store))
    return list(spec)


def tool_schemas(store: ClinicalStore) -> list[dict]:
    """OpenAI-style tool definitions with enums drawn from the current store.

    Regenerated on every turn. After a rename, the old id is gone from the
    schema — which is the harness's approximation of the mask, and the reason
    the two arms are comparable at all.
    """
    tools = []
    for name, method in MANIFEST.items():
        properties: dict[str, dict] = {}
        for arg in method.args:
            values = legal_values(store, name, arg)
            if values is None:
                properties[arg] = {"type": "string", "description": _describe(name, arg)}
            else:
                properties[arg] = {"type": "string", "enum": values}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": method.summary,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(method.args),
                    "additionalProperties": False,
                },
            },
        })
    return tools


def _describe(method: str, arg: str) -> str:
    if method == "rename" and arg == "name":
        return "The new name. Letters, digits, spaces, dots, dashes, underscores."
    if method == "set_metadata" and arg == "value":
        allowed = "; ".join(f"{k}: {'|'.join(v)}" for k, v in METADATA_VALUES.items())
        return f"The value. Controlled vocabularies — {allowed}. Other keys take free text."
    return "A value."


def opcode_strings(store: ClinicalStore, method: str) -> list[str]:
    """Every complete call this method can currently express, as text.

    This is what the trie is built from in the enforced arm. Arguments that are
    free text are emitted as a ``{slot}`` marker for the compiler to turn into a
    trie slot rather than an enumeration.

    Exposed here rather than in the compiler so that both arms provably share
    one definition of "legal".
    """
    method_def = MANIFEST[method]
    combos: list[dict[str, str]] = [{}]

    for arg in method_def.args:
        values = legal_values(store, method, arg)
        nxt = []
        for combo in combos:
            if values is None:
                nxt.append({**combo, arg: "{slot}"})
            else:
                for v in values:
                    nxt.append({**combo, arg: v})
        combos = nxt

    return [
        method + "(" + ", ".join(f"{k}={v}" for k, v in combo.items()) + ")"
        for combo in combos
    ]


def surface_size(store: ClinicalStore) -> dict[str, int]:
    """How many distinct calls each method can express right now.

    Worth watching: enumeration has a ceiling, and a folder with ten thousand
    files would blow it. Reporting the number is how you find out before a demo
    does.
    """
    return {m: len(opcode_strings(store, m)) for m in MANIFEST}
