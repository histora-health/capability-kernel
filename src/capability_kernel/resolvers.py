"""Which records a request is talking about.

This is the input to operand verification, and it is separated from the rule
that uses it for one reason: the rule's logic is settled and the resolver's is
not. String overlap catches the three substitutions measured in the first phase
and will not survive a clinician dictating in Spanish against filenames in
English, which is the likely case in production. When that happens the resolver
gets replaced and the rule does not change.

A resolver answers one question — given a request and the entities that exist,
which ones is it about — and answers it with **candidates rather than a single
best match**. That plural is not caution: a move names what to move *and* where
to put it, so asking for *the* referent picks one and calls the other a
substitution.

Resolvers work over `(id, name)` pairs rather than over store objects, so a
resolver written for one domain works for another. `Odontogram` shares no field
names with `ClinicalStore`, and neither should have to know about this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

#: Words that carry no reference. Matching on these makes everything look like
#: everything — "the study" would score against every study equally.
STOPWORDS = frozenset("""
a an the this that it its of to into out from in on for and or but with without
please can you could would move rename tag set change put file filed place
wrong right new old my our their his her is was are were be been being do does
did done make made take taken get got go went name title label
""".split())


@dataclass(frozen=True)
class Entity:
    """The minimum a resolver needs. Adapted from whatever the domain holds."""

    id: str
    name: str


@dataclass
class Reference:
    """One entity the request may be about, and how strongly."""

    entity: Entity
    score: float
    #: The words that matched, so a person reading a warning can see why.
    matched: tuple[str, ...] = ()


class Resolver(Protocol):
    """Given a request and what exists, which entities it refers to."""

    def candidates(self, entities: Iterable[Entity],
                   message: str) -> list[Reference]:
        """Best first. Empty when the request names nothing."""
        ...


class LexicalResolver:
    """String overlap between an entity's name and the words of the request.

    It suffices for the measured cases because someone asking for "the perio
    chart" uses the words that are in `perio_chart.pdf`. Four details, each
    found by a failing test rather than by reasoning.

    **Name and id are scored separately** rather than pooled, because pooling
    let `std` and `hyg` dilute a study literally called *Hygiene* below
    threshold.

    **Words naming more than one entity are ignored**, because identifier
    conventions manufacture them: `std_hyg`, `std_endo` and `std_ortho` all
    contribute "std", so mentioning any study made every study a candidate —
    which is how an injection naming `std_hyg` got a rename of `std_ortho`
    treated as legitimate.

    **Matching is by prefix from four characters**, because clinicians write
    "panoramic" and the file is `pano_march.dcm`. Exact equality missed that and
    fired on legitimate work, which is the failure mode that gets a guard
    switched off.

    **And the result is plural**, for the reason in the module docstring.
    """

    def __init__(self, min_score: float = 0.34, prefix_from: int = 4) -> None:
        self.min_score = min_score
        self.prefix_from = prefix_from

    def candidates(self, entities: Iterable[Entity],
                   message: str) -> list[Reference]:
        entities = list(entities)
        words = self._words(message)
        ambient = self._ambient(entities)

        scored = []
        for entity in entities:
            # Name and id are two ways of referring to one thing, so an entity
            # is scored on whichever matches better rather than on both pooled.
            score, matched = max(
                (self._score(self._words(entity.name) - ambient, words),
                 self._score(self._words(entity.id) - ambient, words)),
                key=lambda pair: pair[0])
            if score >= self.min_score:
                scored.append(Reference(entity, score, matched))

        return sorted(scored, key=lambda r: -r.score)

    # ── internals ────────────────────────────────────────────────────────────

    def _words(self, text: str) -> set[str]:
        """Content words, split on anything that is not a letter or digit.

        Splitting this way is what makes `perio_chart.pdf` match "perio chart":
        the identifier and the sentence become the same two words.

        Short tokens are dropped as noise — except numeric ones, which are the
        opposite of noise. FDI notation names a tooth in two digits, and the
        length filter made every tooth invisible to the resolver: a dictation
        saying "el 36" resolved to nothing, so the operand rule allowed a
        procedure recorded on any tooth at all. Found by running the rule on the
        second domain, which is what the second domain is for.
        """
        return {w for w in re.split(r"[^a-z0-9]+", text.lower())
                if w and w not in STOPWORDS and (len(w) > 2 or w.isdigit())}

    def _ambient(self, entities: list[Entity]) -> set[str]:
        """Words that name more than one entity, and so name none of them."""
        seen: dict[str, int] = {}
        for entity in entities:
            for word in self._words(entity.name) | self._words(entity.id):
                seen[word] = seen.get(word, 0) + 1
        return {w for w, n in seen.items() if n > 1}

    def _matches(self, a: str, b: str) -> bool:
        return a == b or (min(len(a), len(b)) >= self.prefix_from
                          and (a.startswith(b) or b.startswith(a)))

    def _score(self, entity_words: set[str],
               message_words: set[str]) -> tuple[float, tuple[str, ...]]:
        if not entity_words:
            return 0.0, ()
        shared = {e for e in entity_words
                  if any(self._matches(e, m) for m in message_words)}
        # Fraction of the *entity's* words present, not of the message's. A
        # long request should not dilute a short filename it names exactly.
        return len(shared) / len(entity_words), tuple(sorted(shared))


#: Where an embedding resolver goes when the lexical one stops being enough.
#:
#: The expected trigger is cross-language reference — a clinician dictating in
#: Spanish against filenames in English — which string overlap cannot reach at
#: any threshold. `EmbeddingGemma` through sentence-transformers is the
#: candidate: same model family, local, and much cheaper than an LLM call.
#:
#: Not written yet, deliberately. The lexical resolver catches every case
#: measured so far, and a resolver added before it is needed is a resolver whose
#: failure modes nobody has seen.
DEFAULT = LexicalResolver()
