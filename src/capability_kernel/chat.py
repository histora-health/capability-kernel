"""The enforced arm: a chat loop where the mask is the only gate.

Structurally the mirror of :mod:`harness`, and deliberately so — same store,
same manifest, same prompts. The one difference is where legality is decided.
The harness checks a call after the model produced it; here the call could not
have been produced.

Two things the mask does not do, both found by running it and both bounded
here rather than papered over.

**It does not make an action useful.** Denied what it wanted, gemma4 looped on a
legal rename that changed nothing, five times. Enforcement guarantees the action
is in the surface, never that it is worth taking, so the loop counts no-ops and
stops.

**It does not make an action correct.** Asked to tag a file, the model tagged
the study it sits in. Both are legal opcodes and the mask cannot tell them
apart — that failure needs a better model or a better prompt, and saying so is
more useful than pretending the mechanism covers it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .compiler import ARM, CLOSE, CompiledSurface, compile_surface
from .manifest import MANIFEST, VIRTUAL
from .processor import CapabilityProcessor, Telemetry
from .store import ClinicalStore, StoreError

SYSTEM = """\
You manage a patient's clinical folder. It has studies (folders) and files
inside them, and both carry metadata.

To act, write a line of exactly this form and nothing else on that line:

ACTION <method> <arg>=<value> ...

The available methods are decline, rename, move and set_metadata. Only the
entities listed below exist. A signed study is closed: it and everything inside
it cannot be changed, and it is not listed as a valid target.

If what you were asked cannot be done — the target is closed, or no method
does it — write:

ACTION decline reason=<why>

Never substitute a different target for one you cannot act on. Take one action
at a time. When the request is satisfied, reply in plain words.
"""


@dataclass
class Executed:
    method: str
    args: dict
    result: str
    changed: bool


@dataclass
class EnforcedTurn:
    text: str = ""
    actions: list[Executed] = field(default_factory=list)
    #: Actions the store still refused. Under a correct surface this stays
    #: empty — it is the check that the compiler and the store agree, not a
    #: safety net the design relies on.
    refused: list[str] = field(default_factory=list)
    no_ops: int = 0
    latency_s: float = 0.0


class EnforcedChat:
    """Chat over a store where every action is generated under the mask.

    :param max_actions: hard bound per user message.
    :param max_no_ops: consecutive actions that change nothing before the loop
        gives up. Two, because one repeat is a coincidence and the observed
        failure repeated five times.
    """

    def __init__(self, store: ClinicalStore, llama, *,
                 max_actions: int = 6, max_no_ops: int = 2,
                 max_tokens: int = 256, temperature: float = 0.0) -> None:
        self.store = store
        self.llama = llama
        self.max_actions = max_actions
        self.max_no_ops = max_no_ops
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.telemetry = Telemetry()
        self.transcript: list[tuple[str, str]] = []
        self._surface: CompiledSurface | None = None

    # ── Tokenizer plumbing ───────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        return self.llama.tokenize(text.encode(), add_bos=False, special=False)

    def _detokenize(self, tokens: list[int]) -> str:
        return self.llama.detokenize(list(tokens)).decode("utf-8", "replace")

    def _recompile(self) -> CompiledSurface:
        """Rebuild the surface from the world as it is now.

        Called after every executed action, which is the whole point: the trie
        is a projection of state. A file that just moved into a signed study
        stops being movable on the next token, not on the next turn.
        """
        self._surface = compile_surface(self.store, self._tokenize)
        return self._surface

    # ── Prompting ────────────────────────────────────────────────────────────

    def _prompt(self, user_message: str) -> str:
        lines = [SYSTEM, "", "Current folder:", self.store.describe(), ""]
        for role, text in self.transcript:
            lines.append(f"{role}: {text}")
        lines.append(f"User: {user_message}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # ── The loop ─────────────────────────────────────────────────────────────

    def send(self, user_message: str) -> EnforcedTurn:
        from llama_cpp import LogitsProcessorList

        turn = EnforcedTurn()
        started = time.time()
        prompt = self._prompt(user_message)
        consecutive_no_ops = 0

        while len(turn.actions) < self.max_actions:
            surface = self._recompile()
            proc = CapabilityProcessor(surface, ARM, self._detokenize,
                                       telemetry=self.telemetry)

            out = self.llama(prompt, max_tokens=self.max_tokens,
                             temperature=self.temperature,
                             stop=[CLOSE + CLOSE, "User:"],
                             logits_processor=LogitsProcessorList([proc]))
            text = out["choices"][0]["text"]

            action = _parse(text)
            if action is None:
                turn.text = _prose(text)
                break

            method, args = action
            executed = self._execute(method, args, turn)
            prompt += text

            if executed is None:
                break
            turn.actions.append(executed)

            if executed.changed:
                consecutive_no_ops = 0
            else:
                consecutive_no_ops += 1
                turn.no_ops += 1
                if consecutive_no_ops >= self.max_no_ops:
                    turn.text = ("Stopped: the last actions changed nothing. "
                                 "What was asked for is probably outside what "
                                 "these methods can do.")
                    break

        self.transcript.append(("User", user_message))
        self.transcript.append(("Assistant", turn.text or _describe(turn.actions)))
        turn.latency_s = time.time() - started
        return turn

    def _execute(self, method: str, args: dict, turn: EnforcedTurn) -> Executed | None:
        if method in VIRTUAL:
            turn.text = args.get("reason", "").strip() or "No action taken."
            return None

        before = self.store.snapshot()
        try:
            result = getattr(self.store, method)(**args)
        except StoreError as exc:
            # Under a correct surface this is unreachable. It is kept because
            # "unreachable" is a claim about the compiler, and a claim worth
            # checking is worth checking loudly.
            turn.refused.append(f"{method}({args}): {exc}")
            return None
        return Executed(method=method, args=args, result=result,
                        changed=self.store.snapshot() != before)


def _parse(text: str) -> tuple[str, dict] | None:
    """Read back an action the mask produced.

    Not validation. Every token here came through the trie, so the shape is
    guaranteed; this only has to recover the fields.
    """
    for line in text.splitlines():
        line = line.strip()
        if ARM not in line:
            continue
        line = line.split(ARM, 1)[1].strip()
        parts = line.split()
        if not parts or parts[0] not in MANIFEST:
            continue
        method, args = parts[0], {}
        key = None
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                args[key] = value
            elif key:
                args[key] += " " + part      # a slot value with spaces in it
        if set(args) == set(MANIFEST[method].args):
            return method, args
    return None


def _prose(text: str) -> str:
    """Everything the model said that was not an action."""
    kept = [ln for ln in text.splitlines() if ARM not in ln]
    return "\n".join(kept).strip().lstrip("<|channel>thought").strip()


def _describe(actions: list[Executed]) -> str:
    return "; ".join(a.result for a in actions) if actions else "(no action)"
