"""
Language metrics beyond sentiment:
  - Average sentence length (words)
  - Lexical diversity (type-token ratio)
  - Flesch reading ease and Flesch-Kincaid grade level
  - Term density (per 1 000 words) for violence, uncertainty, and emotional vocabulary
  - Raw word count
"""

import re

import textstat

# ── Term vocabularies ──────────────────────────────────────────────────────────

VIOLENCE_TERMS: set[str] = {
    "war", "kill", "killed", "killing", "death", "deaths", "dead", "murder",
    "murdered", "attack", "attacks", "attacked", "destroy", "destroyed",
    "violence", "violent", "combat", "battle", "battles", "massacre",
    "massacred", "genocide", "casualties", "casualty", "wounded", "conflict",
    "conflicts", "fight", "fighting", "fought", "soldier", "soldiers",
    "army", "armies", "military", "bomb", "bombs", "bombed", "weapon",
    "weapons", "bloodshed", "atrocity", "atrocities", "execution", "siege",
    "assassination", "assassinated", "torture", "brutality",
}

UNCERTAINTY_TERMS: set[str] = {
    "alleged", "allegedly", "reportedly", "claimed", "claims", "disputed",
    "disputes", "unclear", "unknown", "uncertain", "uncertainty",
    "controversial", "controversy", "debate", "debated", "questioned",
    "doubt", "doubtful", "unverified", "possible", "possibly", "may",
    "might", "perhaps", "supposedly", "purportedly", "ambiguous",
    "speculated", "speculation", "contested",
}

EMOTIONAL_TERMS: set[str] = {
    "tragedy", "tragic", "devastating", "devastation", "horrific", "horrible",
    "terrible", "wonderful", "remarkable", "extraordinary", "celebrated",
    "beloved", "revered", "condemned", "feared", "admired", "notorious",
    "infamous", "legendary", "heroic", "hero", "villain", "villainous",
    "glorious", "glory", "shameful", "shame", "inspiring", "inspiration",
    "courageous", "courage", "brave", "bravery", "coward", "cowardice",
}


# ── Core metrics ───────────────────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def average_sentence_length(text: str) -> float:
    sentences = _sentences(text)
    if not sentences:
        return 0.0
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    return sum(lengths) / len(lengths)


def lexical_diversity(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def term_density(text: str, terms: set[str]) -> float:
    """Count of matching terms per 1 000 words."""
    words = _words(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in terms)
    return count / len(words) * 1000


def compute_all_metrics(text: str) -> dict[str, float]:
    return {
        "word_count": len(_words(text)),
        "avg_sentence_length": round(average_sentence_length(text), 2),
        "lexical_diversity": round(lexical_diversity(text), 4),
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "violence_density": round(term_density(text, VIOLENCE_TERMS), 3),
        "uncertainty_density": round(term_density(text, UNCERTAINTY_TERMS), 3),
        "emotional_density": round(term_density(text, EMOTIONAL_TERMS), 3),
    }


def compute_group_metrics(
    texts: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compute metrics for every article in a dict keyed by title."""
    return {title: compute_all_metrics(text) for title, text in texts.items()}
