"""Distributed tracing for multi-agent pipelines — Langfuse integration."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class TracingManager:
    """
    Manages distributed traces across the multi-agent pipeline.

    All methods are safe to call even if Langfuse is unavailable or
    the API has changed — tracing failures never crash the pipeline.
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
        if not self.enabled or not self._langfuse:
            return name
        try:
            trace = self._langfuse.trace(name=name, metadata=metadata or {})
            self._traces[name] = trace
            return trace.id
        except Exception as e:
            logger.debug("tracing.start_trace_failed", error=str(e))
            return name

    def start_span(self, trace_name: str, span_name: str, metadata: dict[str, Any] | None = None) -> Any:
        if not self.enabled or not self._langfuse:
            return None
        try:
            trace = self._traces.get(trace_name)
            if trace:
                return trace.span(name=span_name, metadata=metadata or {})
        except Exception as e:
            logger.debug("tracing.start_span_failed", error=str(e))
        return None

    def end_span(self, span: Any, output: dict[str, Any] | None = None) -> None:
        if span is None:
            return
        try:
            span.end(output=output)
        except Exception:
            pass

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
        if not self.enabled or not self._langfuse:
            return
        try:
            trace = self._traces.get(trace_name)
            if trace:
                trace.generation(
                    name=name,
                    model=model,
                    input=input_text,
                    output=output_text,
                    usage=usage,
                    metadata=metadata or {},
                )
        except Exception as e:
            logger.debug("tracing.log_generation_failed", error=str(e))

    def flush(self) -> None:
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception:
                pass
