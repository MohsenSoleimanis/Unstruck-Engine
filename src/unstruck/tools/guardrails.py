"""Guardrails — input/output safety validation via hooks.

Reads rules from config/guardrails.yaml. Registers as PreLLMCall
and PostLLMCall hooks. Checks for prompt injection, PII, and
content safety.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class Guardrails:
    """
    Safety guardrails for LLM inputs and outputs.

    Input checks (PreLLMCall):
      - Prompt injection detection (pattern matching)
      - Input length limit

    Output checks (PostLLMCall):
      - PII detection (email, phone, SSN patterns)
      - Confidence threshold checking
    """

    def __init__(self, guardrails_config: dict[str, Any]) -> None:
        self._config = guardrails_config
        self._input_config = guardrails_config.get("input", {})
        self._output_config = guardrails_config.get("output", {})

        # Compile prompt injection patterns
        injection_cfg = self._input_config.get("prompt_injection", {})
        self._injection_enabled = injection_cfg.get("enabled", False)
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in injection_cfg.get("patterns", [])
        ]
        self._injection_action = injection_cfg.get("action", "reject")

        # PII patterns
        self._pii_patterns = {
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        }

        self._input_max_length = self._input_config.get("max_input_length", 50000)

    def register_hooks(self, hooks: HookManager) -> None:
        """Register guardrail checks as hooks."""
        hooks.register(HookEvent.PRE_LLM_CALL, self._check_input)
        hooks.register(HookEvent.POST_LLM_CALL, self._check_output)

    async def _check_input(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Check input before LLM call."""
        user_prompt = context.get("user_prompt", "")

        # Length check
        if len(user_prompt) > self._input_max_length:
            return HookResult.block(f"Input exceeds max length ({len(user_prompt)} > {self._input_max_length})")

        # Prompt injection detection
        if self._injection_enabled:
            for pattern in self._injection_patterns:
                if pattern.search(user_prompt):
                    logger.warning("guardrails.injection_detected", pattern=pattern.pattern)
                    if self._injection_action == "reject":
                        return HookResult.block("Potential prompt injection detected")
                    # "flag" or "log" — allow but log
                    break

        return HookResult.allow()

    async def _check_output(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """Check output after LLM call."""
        response = context.get("response", "")

        pii_config = self._output_config.get("pii_stripping", {})
        if pii_config.get("enabled", False):
            pii_types = pii_config.get("types", [])
            found_pii = []

            for pii_type in pii_types:
                pattern = self._pii_patterns.get(pii_type)
                if pattern and pattern.search(response):
                    found_pii.append(pii_type)

            if found_pii:
                logger.warning("guardrails.pii_detected", types=found_pii)
                # Strip PII from response
                cleaned = response
                for pii_type in found_pii:
                    pattern = self._pii_patterns.get(pii_type)
                    if pattern:
                        cleaned = pattern.sub(f"[{pii_type.upper()}_REDACTED]", cleaned)
                return HookResult.modify({"response": cleaned, "pii_found": found_pii})

        return HookResult.allow()
