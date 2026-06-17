from transformers import pipeline

from src.utils import chunk_text

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _classifier


def score_to_float(result: dict) -> float:
    """Map {'label': 'POSITIVE'|'NEGATIVE', 'score': float} to [-1, 1]."""
    return result["score"] if result["label"] == "POSITIVE" else -result["score"]


def analyze(text: str, chunk_size: int = 400) -> list[dict]:
    """Chunk text and return one result dict per chunk."""
    clf = _get_classifier()
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
