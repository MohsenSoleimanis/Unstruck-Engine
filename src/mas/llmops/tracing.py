"""Distributed tracing for multi-agent pipelines — Langfuse + OpenTelemetry."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class TracingManager:
    """
    Manages distributed traces across the multi-agent pipeline.

    Integrates with:
      - Langfuse for LLM-specific tracing (prompts, completions, costs)
      - OpenTelemetry for general distributed tracing (spans, events)
    """

    def __init__(
        self,
        enabled: bool = True,
        langfuse_public_key: str = "",
        langfuse_secret_key: str = "",
        langfuse_host: str = "https://cloud.langfuse.com",
    ):
        self.enabled = enabled
        self._langfuse = None
        self._traces: dict[str, Any] = {}

        if enabled and langfuse_public_key:
            self._init_langfuse(langfuse_public_key, langfuse_secret_key, langfuse_host)

    def _init_langfuse(self, public_key: str, secret_key: str, host: str) -> None:
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("tracing.langfuse_connected")
        except Exception as e:
            logger.warning("tracing.langfuse_fallback", error=str(e))

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Start a new trace (e.g., for a user request or orchestrator run)."""
        if not self.enabled:
            return name

        if self._langfuse:
            trace = self._langfuse.trace(name=name, metadata=metadata or {})
            self._traces[name] = trace
            return trace.id

        logger.info("trace.start", name=name)
        return name

    def start_span(
        self,
        trace_name: str,
        span_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Start a span within a trace (e.g., for an agent execution)."""
        if not self.enabled:
            return None

        trace = self._traces.get(trace_name)
        if trace and self._langfuse:
            return trace.span(name=span_name, metadata=metadata or {})

        logger.debug("span.start", trace=trace_name, span=span_name)
        return None

    def end_span(self, span: Any, output: dict[str, Any] | None = None) -> None:
        if span and hasattr(span, "end"):
            span.end(output=output)

    def log_generation(
        self,
        trace_name: str,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM generation event."""
        trace = self._traces.get(trace_name)
        if trace and self._langfuse:
            trace.generation(
                name=name,
                model=model,
                input=input_text,
                output=output_text,
                usage=usage,
                metadata=metadata or {},
            )
        else:
            logger.debug("generation.log", name=name, model=model)

    def flush(self) -> None:
        if self._langfuse:
            self._langfuse.flush()
