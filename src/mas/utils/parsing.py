"""Shared LLM output parsing utilities — eliminates DRY violations across agents."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def extract_json(raw: str) -> Any:
    """
    Extract and parse JSON from LLM output.

    Handles:
      - Raw JSON
      - JSON wrapped in markdown fences (```json ... ```)
      - Multiple fenced blocks (takes the first valid one)
    """
    text = raw.strip()

    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    match = _FENCED_JSON_RE.search(text)
    if match:
        return json.loads(match.group(1).strip())

    raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)


def extract_token_usage(response: Any) -> dict[str, int]:
    """
    Extract token usage from a LangChain LLM response.

    Returns a dict with input_tokens and output_tokens, or empty dict
    if usage metadata is not available.
    """
    metadata = getattr(response, "usage_metadata", None)
    if not metadata:
        return {}
    return {
        "input_tokens": metadata.get("input_tokens", 0),
        "output_tokens": metadata.get("output_tokens", 0),
    }
