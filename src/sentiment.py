"""
Sentiment and emotion analysis.

Sentiment  : siebert/sentiment-roberta-large-english
             Binary POSITIVE / NEGATIVE, trained on diverse English sources
             (news, reviews, social media) — better fit for Wikipedia than SST-2.

Emotion    : j-hartmann/emotion-english-distilroberta-base
             7-class: anger, disgust, fear, joy, neutral, sadness, surprise.
"""

from __future__ import annotations

from transformers import pipeline

from src.utils import chunk_text

_sentiment_clf = None
_emotion_clf = None


def _get_sentiment_clf():
    global _sentiment_clf
    if _sentiment_clf is None:
        _sentiment_clf = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            truncation=True,
            max_length=512,
        )
    return _sentiment_clf


def _get_emotion_clf():
    global _emotion_clf
    if _emotion_clf is None:
        _emotion_clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            truncation=True,
            max_length=512,
            top_k=None,
        )
    return _emotion_clf


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_to_float(result: dict) -> float:
    """Map {'label': 'POSITIVE'|'NEGATIVE', 'score': float} to [-1, 1]."""
    return result["score"] if result["label"] == "POSITIVE" else -result["score"]


# ── Sentiment ──────────────────────────────────────────────────────────────────

def analyze(text: str, chunk_size: int = 400) -> list[dict]:
    """Chunk text and return one result dict per chunk."""
    clf = _get_sentiment_clf()
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return []
    return clf(chunks, batch_size=16)


def analyze_sections(sections: dict[str, str]) -> dict[str, list[dict]]:
    return {title: analyze(text) for title, text in sections.items()}


def average_score(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(score_to_float(r) for r in results) / len(results)


def section_averages(section_results: dict[str, list[dict]]) -> dict[str, float]:
    return {title: average_score(results) for title, results in section_results.items()}


# ── Emotion ────────────────────────────────────────────────────────────────────

def analyze_emotions(text: str, chunk_size: int = 400) -> dict[str, float]:
    """Return average score per emotion label across all chunks of `text`.

    Labels: anger, disgust, fear, joy, neutral, sadness, surprise.
    """
    clf = _get_emotion_clf()
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return {}

    # clf returns list[list[dict]] — one list of {label, score} per chunk
    all_results: list[list[dict]] = clf(chunks, batch_size=8)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk_results in all_results:
        for item in chunk_results:
            label = item["label"]
            totals[label] = totals.get(label, 0.0) + item["score"]
            counts[label] = counts.get(label, 0) + 1

    return {label: round(totals[label] / counts[label], 4) for label in totals}


def emotion_profile(texts: dict[str, str]) -> dict[str, dict[str, float]]:
    """Compute emotion averages for a collection of articles."""
    return {title: analyze_emotions(text) for title, text in texts.items()}
