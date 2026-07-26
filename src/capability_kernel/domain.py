"""A domain: what may be done, and where the legal values come from.

The first phase had one manifest, hard-coded, with the phase rule — an
unrecorded change admits nothing but recording it — written into the function
that computed it. That is fine for one domain and wrong for two, and two is the
point: procedure coding and study filing have different methods, different
argument sources and different notions of what phase the world is in.

So a domain is a value. It carries its methods, its phase function, and which of
its methods have no store operation behind them. Everything the option surface
does is a method on it, computed from a store passed in rather than from a store
it owns — which is what lets two domains share one process, and what keeps
`Domain` from assuming the clinical store's shape.

Nothing here decides whether an action is *permitted*. That is the firmware
layer's job. This decides what the model may **propose**, which is a different
question and the reason the two are separate: a surface that never offers an
action produces no rejection to recover from, and recovering from rejections is
what a small local model cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


def _arity(source) -> int:
    import inspect
    try:
        return len(inspect.signature(source).parameters)
    except (TypeError, ValueError):
        return 1


def _apply(source, store, chosen: dict):
    return source(store, chosen) if _arity(source) > 1 else source(store)


@dataclass(frozen=True)
class Method:
    """One operation, and where each argument's legal values come from.

    A value source is a callable when the values are state — which is most of
    the time and the whole point — a tuple when the vocabulary is fixed, or
    ``None`` for free text.

    A callable takes ``(store)`` or ``(store, chosen)``, where `chosen` is the
    arguments decided so far. The second form exists because arguments chain:
    which surfaces a tooth has depends on the tooth, and which codes are valid
    depends on the surface. Computing each argument independently produces
    combinations that do not exist — an occlusal surface on an incisor — and
    the alternative, asking the model once per argument, is three round trips
    for something a dentist does several times per consultation.
    """

    name: str
    summary: str
    args: dict[str, Callable | tuple | None]

    def describe(self, arg: str) -> str:
        return self.arg_help.get(arg, "A value.")

    #: Per-argument prose for the tool schema. Separate from `summary` because a
    #: model reads the argument description while choosing the argument.
    arg_help: dict[str, str] = field(default_factory=dict)


def _label(args: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def always_all(store) -> None:
    """The default phase: every method exists all the time.

    Returning None rather than the method list keeps `Domain` from having to
    know its own method names here, and reads as "no phase constraint" rather
    than as a list that happens to be complete.
    """
    return None


@dataclass(frozen=True)
class Domain:
    """A declaration of what an agent may propose in this domain."""

    name: str
    methods: dict[str, Method]
    #: Given a store, which methods exist right now. None means all of them.
    phase: Callable[[object], tuple[str, ...] | None] = always_all
    #: Methods with no store operation behind them — completing one changes
    #: nothing. `decline` is the archetype: refusing has to be expressible, and
    #: a surface containing only ways to act can only be satisfied by acting.
    virtual: tuple[str, ...] = ()
    #: ``store -> Iterable[Entity]`` for operand verification. Here rather than
    #: in the resolver because only the domain knows how its records are named,
    #: and a resolver that knew would work for one domain only.
    nameable: Callable[[object], list] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.virtual) - set(self.methods)
        if unknown:
            raise ValueError(
                f"{self.name}: virtual names {sorted(unknown)} are not methods")

    # ── The option surface ───────────────────────────────────────────────────

    def enabled_methods(self, store) -> tuple[str, ...]:
        """Which methods have a path right now.

        The phase function may narrow this to a subset, and may narrow it to
        something that excludes `decline` — which is a real thing to allow. A
        state you can decline your way out of permits exactly what the phase
        rule exists to prevent, and the clinical audit rule depends on it.
        """
        allowed = self.phase(store)
        return tuple(self.methods) if allowed is None else tuple(allowed)

    def legal_values(self, store, method: str, arg: str,
                     chosen: dict | None = None) -> list[str] | None:
        """What `arg` may be, given what has already been chosen.

        `chosen` is empty for an independent argument and carries the earlier
        choices for a dependent one. Passing it always, and letting sources
        that do not need it ignore it, keeps the two kinds from being different
        types of thing.
        """
        source = self.methods[method].args[arg]
        if source is None:
            return None
        if not callable(source):
            return list(source)
        return list(_apply(source, store, chosen or {}))

    def is_chained(self, method: str) -> bool:
        """Whether any argument of this method depends on another.

        Worth asking before enumerating: a chained method's combinations are a
        tree rather than a product, and a caller that assumes otherwise
        silently produces options that do not exist.
        """
        return any(callable(src) and _arity(src) > 1
                   for src in self.methods[method].args.values())

    #: Above this many combinations a chained method is not offered as a single
    #: choice. Enumeration has a ceiling and the honest failure is a loud one:
    #: a schema with ten thousand enum values is a prompt nobody can read and a
    #: latency nobody budgeted.
    MAX_COMBINATIONS = 400

    def tool_schemas(self, store) -> list[dict]:
        """OpenAI-style tool definitions for this turn.

        Regenerated every call. After a study is signed it is gone from the
        enums, which is the mechanism — not a rule that says it may not be
        touched, but an option that is not there.

        A **chained** method cannot be expressed this way argument by argument:
        which surfaces exist depends on the tooth, and a JSON Schema has no way
        to say so. Computing each enum independently would offer an occlusal
        surface on an incisor. So a chained method is offered as a single
        enumerated choice over the combinations that actually exist — the
        program computes them, the model picks one, and the result is valid by
        construction in one call rather than three.
        """
        tools = []
        for name in self.enabled_methods(store):
            method = self.methods[name]

            if self.is_chained(name):
                tools.append(self._chained_schema(store, name, method))
                continue

            properties = {}
            for arg in method.args:
                values = self.legal_values(store, name, arg)
                properties[arg] = ({"type": "string", "enum": values} if values
                                   is not None else
                                   {"type": "string", "description": method.describe(arg)})
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": method.summary,
                    "parameters": {"type": "object", "properties": properties,
                                   "required": list(method.args),
                                   "additionalProperties": False},
                },
            })
        return tools

    def _chained_schema(self, store, name: str, method: Method) -> dict:
        options = self.combinations(store, name)
        if len(options) > self.MAX_COMBINATIONS:
            raise ValueError(
                f"{name} has {len(options)} valid combinations, over the ceiling "
                f"of {self.MAX_COMBINATIONS}. Narrow the surface before offering "
                f"it — an unreadable schema is a worse failure than a refused one.")

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": (f"{method.summary} Choose one of the "
                                f"{len(options)} combinations that exist."),
                "parameters": {
                    "type": "object",
                    "properties": {"choice": {
                        "type": "string",
                        "enum": [_label(c) for c in options],
                        "description": "One of the enumerated combinations.",
                    }},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        }

    def combinations(self, store, method: str) -> list[dict[str, str]]:
        """Every complete argument set this method can currently express.

        A tree rather than a product: each argument's options are recomputed
        against the choices before it, which is what keeps combinations that do
        not exist out of the list.
        """
        combos: list[dict[str, str]] = [{}]
        for arg in self.methods[method].args:
            grown = []
            for chosen in combos:
                values = self.legal_values(store, method, arg, chosen)
                grown += [{**chosen, arg: v}
                          for v in (values if values is not None else ["{slot}"])]
            combos = grown
        return combos

    def parse_choice(self, method: str, choice: str) -> dict[str, str]:
        """Turn a chosen label back into arguments.

        The model returns the label it was offered, so this is recovery rather
        than validation — the value came from a list the program wrote.
        """
        args = {}
        for part in choice.split(", "):
            key, _, value = part.partition("=")
            args[key.strip()] = value.strip()
        return args

    # ── Enumeration, for inspecting the surface and measuring the ceiling ────

    def action_strings(self, store, method: str) -> list[str]:
        """Every complete call this method can currently express.

        Free-text arguments become a ``{slot}`` marker. For reading the surface
        and for `surface_size`; nothing on the production path needs it, since
        the model is given tool schemas rather than a list of strings.

        (It used to be called ``opcode_strings``. An opcode was a path through
        the token trie, and there is no trie on this path any more.)
        """
        return [f"{method}({_label(c)})"
                for c in self.combinations(store, method)]

    def surface_size(self, store) -> dict[str, int]:
        """How many distinct calls each method can express right now.

        Worth watching rather than assuming. A demo folder has seven entities
        and a real clinical history has hundreds, and enumeration has a ceiling
        that has been named and not measured.
        """
        return {m: len(self.action_strings(store, m)) for m in self.methods}
