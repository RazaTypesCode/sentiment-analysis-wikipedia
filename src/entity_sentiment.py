"""
Entity-level sentiment analysis.

1. Use spaCy to extract named entities (PERSON, ORG, GPE, LOC, EVENT).
2. For each entity, collect all context windows (±window chars around each mention).
3. Run the sentiment classifier on the combined context and return a score per entity.
"""

from __future__ import annotations

from collections import Counter

import spacy

from src.sentiment import analyze, average_score

_nlp = None

ENTITY_TYPES = ("PERSON", "ORG", "GPE", "LOC", "EVENT")


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model not found. Run:  python -m spacy download en_core_web_sm"
            )
    return _nlp


def extract_entities(
    text: str,
    types: tuple[str, ...] = ENTITY_TYPES,
    max_chars: int = 500_000,
) -> dict[str, list[str]]:
    """Return {entity_label: [entity_text, ...]} for all detected entities."""
    nlp = _get_nlp()
    doc = nlp(text[:max_chars])
    result: dict[str, list[str]] = {}
    for ent in doc.ents:
        if ent.label_ in types:
            result.setdefault(ent.label_, []).append(ent.text)
    return result


def _collect_contexts(text: str, entity: str, window: int) -> list[str]:
    """All context windows around every mention of `entity` in `text`."""
    contexts: list[str] = []
    text_lower = text.lower()
    needle = entity.lower()
    start = 0
    while True:
        idx = text_lower.find(needle, start)
        if idx == -1:
            break
        lo = max(0, idx - window)
        hi = min(len(text), idx + len(entity) + window)
        contexts.append(text[lo:hi])
        start = idx + 1
    return contexts


def entity_sentiment(text: str, entity: str, window: int = 300) -> float:
    """Average sentiment score in [-1, 1] for all contexts mentioning `entity`."""
    contexts = _collect_contexts(text, entity, window)
    if not contexts:
        return 0.0
    combined = " ".join(contexts)
    results = analyze(combined)
    return average_score(results)


def analyze_top_entities(
    text: str,
    top_n: int = 12,
    window: int = 300,
    types: tuple[str, ...] = ENTITY_TYPES,
) -> dict[str, float]:
    """Return {entity_text: sentiment_score} for the top_n most frequent entities."""
    nlp = _get_nlp()
    doc = nlp(text[:500_000])

    counts = Counter(
        ent.text
        for ent in doc.ents
        if ent.label_ in types and len(ent.text) > 2
    )
    top_entities = [e for e, _ in counts.most_common(top_n)]

    return {ent: entity_sentiment(text, ent, window) for ent in top_entities}
