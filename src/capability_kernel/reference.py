"""Did the model act on the record the user was talking about?

This is the guard for the failure that survived everything else. Told to act on
a record it may not touch, a model performs a legal action on a *different*
record and reports success — measured at 5 of 20 on gemma4:12b and 3 of 12 on
gemma-4-E4B, and triggered by phrasing that sounds like ordinary filing rather
than by attack.

Nothing upstream catches it. The action is inside the capability surface, so a
mask permits it; the arguments are all in their enums, so a validator permits it;
and no violation is logged because nothing was violated. The record is simply
wrong.

What does catch it is comparing the emitted target against what the request was
about. That comparison needs no model, no mask and no interpretability — it is
string overlap between the entity's name and the user's words, which is enough
because a clinician asking about "the perio chart" uses the words that are in
`perio_chart.pdf`.

Deliberately conservative. It reports a *mismatch* for a human to resolve rather
than choosing a target itself: a guard that silently corrected the model would
be making clinical decisions by string similarity, which is worse than the
failure it prevents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import ClinicalStore, Entity

#: Words that carry no reference. Matching on these makes everything look like
#: everything — "the study" would score against all three studies equally.
STOPWORDS = frozenset("""
a an the this that it its of to into out from in on for and or but with without
please can you could would move rename tag set change put file filed place put
wrong right new old my our their his her is was are were be been being do does
did done make made take taken get got go went name title label
""".split())

#: Below this, no entity is considered referred to at all.
MIN_SCORE = 0.34


@dataclass
class Reference:
    """What the message appears to be about, and how sure that is."""

    entity: Entity | None
    score: float
    #: The words that matched, so a person reading the warning can see why.
    matched: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return self.entity is not None and self.score >= MIN_SCORE


def _words(text: str) -> set[str]:
    """Content words, lowercased, split on anything that is not a letter or digit.

    Splitting on non-alphanumerics is what makes `perio_chart.pdf` match "perio
    chart" — the identifier and the sentence become the same two words.
    """
    return {w for w in re.split(r"[^a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def _matches(a: str, b: str) -> bool:
    """Same word, or one a prefix of the other from four characters.

    Clinicians write "panoramic" and the file is `pano_march.dcm`. Exact
    equality misses that, and missing it made the guard fire on legitimate work,
    which is the failure mode that gets a guard switched off.
    """
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _score(entity_words: set[str], message_words: set[str]) -> tuple[float, tuple[str, ...]]:
    if not entity_words:
        return 0.0, ()
    shared = {e for e in entity_words if any(_matches(e, m) for m in message_words)}
    # Fraction of the *entity's* words present, not of the message's. A long
    # request should not dilute a short filename it names exactly.
    return len(shared) / len(entity_words), tuple(sorted(shared))


def candidates(store: ClinicalStore, message: str) -> list[Reference]:
    """Every entity the message plausibly refers to, best first.

    Plural on purpose. A move names two records — what to move and where to put
    it — so asking for *the* referent picks one and calls the other a
    substitution. Measured: "move the panoramic into the endodontics study"
    resolves to endodontics, and a guard built on a single best match then
    blocks a correct move of the panoramic.
    """
    ambient = _ambient(store)
    scored = [r for r in (_reference(store, e, _words(message), ambient)
                          for e in store.nameable()) if r.resolved]
    return sorted(scored, key=lambda r: -r.score)


def _ambient(store: ClinicalStore) -> set[str]:
    """Words that name more than one entity, and so name none of them.

    Identifier conventions produce these by construction: `std_hyg`, `std_endo`
    and `std_ortho` all contribute "std", so a message mentioning any study made
    every study a candidate. Measured on the injection prompt, which names
    `std_hyg` and had the guard treat a rename of `std_ortho` as legitimate
    because both share that prefix.
    """
    seen: dict[str, int] = {}
    for entity in store.nameable():
        for word in _words(entity.name) | _words(entity.id):
            seen[word] = seen.get(word, 0) + 1
    return {w for w, n in seen.items() if n > 1}


def _reference(store: ClinicalStore, entity: Entity, message_words: set[str],
               ambient: set[str]) -> Reference:
    # Name and id are two ways of referring to one thing, so an entity is
    # scored on whichever matches better rather than on both pooled. Pooling
    # dilutes: `std_hyg` contributes "std" and "hyg" as noise, which dropped
    # "the hygiene study" below threshold against a study called Hygiene.
    score, matched = max(
        (_score(_words(entity.name) - ambient, message_words),
         _score(_words(entity.id) - ambient, message_words)),
        key=lambda pair: pair[0])
    return Reference(entity, score, matched)


def resolve(store: ClinicalStore, message: str) -> Reference:
    """The single best referent. See :func:`candidates` for why that is rarely
    what a guard should use."""
    named = candidates(store, message)
    return named[0] if named else Reference(None, 0.0, ())


@dataclass
class Mismatch:
    """The action names a record the request does not.

    `named` is plural because requests usually are: a move mentions what to move
    and where to put it, and reporting only the strongest match would tell a
    reviewer the request was about the destination.
    """

    named: tuple[Entity, ...]
    acted_on: Entity
    matched: tuple[str, ...]

    @property
    def asked_about(self) -> Entity:
        """The strongest match, for callers that want one."""
        return self.named[0]

    def __str__(self) -> str:
        names = ", ".join(e.id for e in self.named)
        return (f"the request refers to {names} — not to {self.acted_on.id!r}, "
                f"which this action would change. Words matched: "
                f"{list(self.matched)}")


def check(store: ClinicalStore, message: str, target: str) -> Mismatch | None:
    """Refuse to execute an action whose target is not what was asked about.

    Returns None when the target matches, when nothing in the message resolves
    to an entity, or when the target simply is the best match. A warning is only
    produced when the message clearly points at one record and the action names
    another — which is exactly the measured failure and nothing else.
    """
    named = candidates(store, message)
    if not named:
        return None                      # nothing was referred to; nothing to contradict

    if any(r.entity is not None and r.entity.id == target for r in named):
        return None                      # the target is one of the records named

    acted = store.get(target)
    if acted is None:
        return None

    matched = tuple(sorted({w for r in named for w in r.matched}))
    return Mismatch(tuple(r.entity for r in named if r.entity), acted, matched)
