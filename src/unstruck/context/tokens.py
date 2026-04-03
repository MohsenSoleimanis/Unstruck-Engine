"""Token counting and truncation — the foundation of context management.

Uses tiktoken for accurate counting. Falls back to character estimation
if tiktoken can't load the model's encoding.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _get_encoder(model: str = "gpt-4o"):
    """Get tiktoken encoder for a model. Cached per model name."""
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model)
    except Exception:
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text. Falls back to len(text) // 4 if tiktoken unavailable."""
    encoder = _get_encoder(model)
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate text to fit within a token budget. Preserves complete words."""
    encoder = _get_encoder(model)
    if encoder is not None:
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoder.decode(tokens[:max_tokens])
    # Fallback: character-based estimate
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
