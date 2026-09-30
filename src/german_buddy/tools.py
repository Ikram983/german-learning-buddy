"""Deterministic tools the agents can call.

These are plain functions — no LLM involved — so their output is always
correct and reproducible. That matters a lot for something like verb
conjugation: you never want an LLM "guessing" a conjugation when a lookup
table/rule can just be right every time.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool

PRONOUNS = ["ich", "du", "er/sie/es", "wir", "ihr", "sie/Sie"]

# A handful of common irregular (strong) present-tense verbs, hardcoded.
# Real irregular conjugation doesn't follow a rule, so this has to be a
# lookup table, not logic. Extend this dict as you add more verbs.
IRREGULAR_VERBS: dict[str, dict[str, str]] = {
    "sein": {"ich": "bin", "du": "bist", "er/sie/es": "ist", "wir": "sind", "ihr": "seid", "sie/Sie": "sind"},
    "haben": {"ich": "habe", "du": "hast", "er/sie/es": "hat", "wir": "haben", "ihr": "habt", "sie/Sie": "haben"},
    "werden": {"ich": "werde", "du": "wirst", "er/sie/es": "wird", "wir": "werden", "ihr": "werdet", "sie/Sie": "werden"},
    "fahren": {"ich": "fahre", "du": "fährst", "er/sie/es": "fährt", "wir": "fahren", "ihr": "fahrt", "sie/Sie": "fahren"},
    "essen": {"ich": "esse", "du": "isst", "er/sie/es": "isst", "wir": "essen", "ihr": "esst", "sie/Sie": "essen"},
    "geben": {"ich": "gebe", "du": "gibst", "er/sie/es": "gibt", "wir": "geben", "ihr": "gebt", "sie/Sie": "geben"},
    "nehmen": {"ich": "nehme", "du": "nimmst", "er/sie/es": "nimmt", "wir": "nehmen", "ihr": "nehmt", "sie/Sie": "nehmen"},
    "sehen": {"ich": "sehe", "du": "siehst", "er/sie/es": "sieht", "wir": "sehen", "ihr": "seht", "sie/Sie": "sehen"},
    "lesen": {"ich": "lese", "du": "liest", "er/sie/es": "liest", "wir": "lesen", "ihr": "lest", "sie/Sie": "lesen"},
    "sprechen": {"ich": "spreche", "du": "sprichst", "er/sie/es": "spricht", "wir": "sprechen", "ihr": "sprecht", "sie/Sie": "sprechen"},
    "laufen": {"ich": "laufe", "du": "läufst", "er/sie/es": "läuft", "wir": "laufen", "ihr": "lauft", "sie/Sie": "laufen"},
    "wissen": {"ich": "weiß", "du": "weißt", "er/sie/es": "weiß", "wir": "wissen", "ihr": "wisst", "sie/Sie": "wissen"},
}

_ENDINGS = {"ich": "e", "du": "st", "er/sie/es": "t", "wir": "en", "ihr": "t", "sie/Sie": "en"}


def _regular_stem(infinitive: str) -> str:
    if not infinitive.endswith("en") and not infinitive.endswith("n"):
        raise ValueError(f"'{infinitive}' doesn't look like a German infinitive (should end in -en/-n)")
    return infinitive[:-2] if infinitive.endswith("en") else infinitive[:-1]


def conjugate_regular(infinitive: str) -> dict[str, str]:
    """Apply the standard weak-verb present-tense rules, including the two
    common spelling adjustments:
      - stem ends in d/t (e.g. 'arbeiten')  -> insert 'e' before -st/-t
      - stem ends in s/ß/z/x (e.g. 'reisen') -> du-form drops the extra s
    """
    stem = _regular_stem(infinitive)
    needs_e = stem.endswith(("d", "t"))
    sibilant = stem.endswith(("s", "ß", "z", "x"))

    forms: dict[str, str] = {}
    for pronoun, ending in _ENDINGS.items():
        e = "e" if needs_e and ending in ("st", "t") else ""
        if pronoun == "du" and sibilant:
            forms[pronoun] = f"{stem}{e}t"  # e.g. reisen -> reist, not reisst
        else:
            forms[pronoun] = f"{stem}{e}{ending}"
    return forms


def conjugate(infinitive: str) -> dict[str, str]:
    """Public entry point: look up irregulars first, fall back to the
    regular rules. This is what gets exposed as a tool to the agents."""
    infinitive = infinitive.strip().lower()
    if infinitive in IRREGULAR_VERBS:
        return dict(IRREGULAR_VERBS[infinitive])
    return conjugate_regular(infinitive)


@tool
def conjugate_verb(infinitive: str) -> str:
    """Conjugate a German verb in the present tense given its infinitive
    form (e.g. 'fahren', 'machen'). Always use this instead of guessing a
    conjugation yourself — irregular verbs especially are easy to get
    wrong from memory."""
    try:
        forms = conjugate(infinitive)
    except ValueError as e:
        return str(e)
    return "\n".join(f"{pronoun}: {form}" for pronoun, form in forms.items())


# --------------------------------------------------------------------------- #
# Mistake tracker — a tiny persisted store the Quiz agent reads from and the
# Correction agent writes to.
# --------------------------------------------------------------------------- #

class MistakeTracker:
    """Counts how often each grammar/vocab topic has tripped you up, so the
    Quiz agent can prioritize what you actually need practice on instead of
    quizzing you randomly."""

    def __init__(self, path: Path):
        self.path = path
        self._counts: Counter[str] = Counter()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._counts = Counter(data)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(self._counts), indent=2), encoding="utf-8")

    def record_mistake(self, topic: str) -> None:
        """topic examples: 'accusative_case', 'verb:fahren', 'der/die/das'."""
        self._counts[topic] += 1
        self._save()

    def top_mistakes(self, n: int = 5) -> list[tuple[str, int]]:
        return self._counts.most_common(n)