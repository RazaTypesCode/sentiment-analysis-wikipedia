def chunk_text(text: str, size: int = 400) -> list[str]:
    """Split text into non-empty chunks of at most `size` characters."""
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def flatten(nested: list[list]) -> list:
    return [item for sublist in nested for item in sublist]
