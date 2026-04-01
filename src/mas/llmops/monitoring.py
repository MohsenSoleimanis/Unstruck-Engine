"""Health monitoring and alerting for the multi-agent system."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


class HealthMonitor:
    """
    Monitors agent health, pipeline throughput, and system status.

    Tracks:
      - Agent availability and failure rates
      - Pipeline latency percentiles
      - Memory utilization
      - Active task count
    """

    def __init__(self) -> None:
        self._agent_status: dict[str, dict[str, Any]] = {}
        self._pipeline_metrics: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []

    def report_agent_status(self, agent_id: str, agent_type: str, healthy: bool, metadata: dict | None = None) -> None:
        self._agent_status[agent_id] = {
            "agent_type": agent_type,
            "healthy": healthy,
            "last_seen": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        if not healthy:
            self._alert(f"Agent {agent_id} ({agent_type}) reported unhealthy", severity="warning")

    def record_pipeline_run(
        self,
        pipeline_id: str,
        duration_ms: int,
        task_count: int,
        success_count: int,
        total_cost: float,
    ) -> None:
        self._pipeline_metrics.append({
            "pipeline_id": pipeline_id,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_ms": duration_ms,
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / max(task_count, 1),
            "total_cost_usd": total_cost,
        })

    def _alert(self, message: str, severity: str = "info") -> None:
        alert = {"message": message, "severity": severity, "timestamp": datetime.utcnow().isoformat()}
        self._alerts.append(alert)
        logger.warning("monitor.alert", **alert)

    def get_health(self) -> dict[str, Any]:
        healthy_agents = sum(1 for s in self._agent_status.values() if s["healthy"])
        total_agents = len(self._agent_status)
        return {
            "status": "healthy" if healthy_agents == total_agents else "degraded",
            "agents": {"total": total_agents, "healthy": healthy_agents},
            "recent_alerts": self._alerts[-10:],
            "pipeline_runs": len(self._pipeline_metrics),
        }

    def get_metrics(self) -> dict[str, Any]:
        if not self._pipeline_metrics:
            return {}
        durations = [m["duration_ms"] for m in self._pipeline_metrics]
        costs = [m["total_cost_usd"] for m in self._pipeline_metrics]
        return {
            "avg_duration_ms": sum(durations) / len(durations),
            "avg_cost_usd": sum(costs) / len(costs),
            "total_runs": len(self._pipeline_metrics),
            "avg_success_rate": sum(m["success_rate"] for m in self._pipeline_metrics) / len(self._pipeline_metrics),
        }
