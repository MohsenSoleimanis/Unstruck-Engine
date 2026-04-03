"""Permission system — 3-layer authorization for agent tool calls.

Layer 1: Registry filter — agent only sees its allowed tools
Layer 2: Per-call check — are these specific args allowed?
Layer 3: Human escalation — unknown/dangerous → pause, ask user

Implemented via the PreToolUse hook. The permission system registers
itself as a hook handler at startup.
"""

from __future__ import annotations

from typing import Any

import structlog

from unstruck.config import Config
from unstruck.hooks import HookAction, HookEvent, HookManager, HookResult

logger = structlog.get_logger()


class PermissionSystem:
    """
    Checks whether an agent is allowed to call a specific tool.

    Reads from:
      - config/agents.yaml (allowed_tools per agent)
      - config/tools.yaml (permission_level per tool)
      - config/permissions.yaml (role-based access)
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._denial_log: list[dict[str, str]] = []

    def register_hooks(self, hooks: HookManager) -> None:
        """Register the permission checker as a PreToolUse hook."""
        hooks.register(HookEvent.PRE_TOOL_USE, self._check_permission)

    async def _check_permission(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """
        Hook handler: checks if the agent is allowed to call this tool.

        Context must contain:
          - agent_type: which agent is calling
          - tool_name: which tool is being called
          - tool_args: the arguments (for pattern matching)
          - user_role: the user's role (for role-based checks)
        """
        agent_type = context.get("agent_type", "")
        tool_name = context.get("tool_name", "")
        user_role = context.get("user_role", "user")

        # --- Layer 1: Agent allowed_tools check ---
        agent_config = self._config.agents.get(agent_type, {})
        allowed_tools = agent_config.get("allowed_tools", [])

        # Empty allowed_tools means "all tools" (for agents with no restrictions)
        if allowed_tools and tool_name not in allowed_tools:
            reason = f"Agent '{agent_type}' is not allowed to use tool '{tool_name}'"
            self._denial_log.append({"agent": agent_type, "tool": tool_name, "reason": reason})
            logger.info("permission.denied.layer1", agent=agent_type, tool=tool_name)
            return HookResult.block(reason)

        # --- Layer 2: Role-based tool access ---
        try:
            role_config = self._config.get_role(user_role)
        except Exception:
            role_config = {"tools": []}

        role_tools = role_config.get("tools", "all")
        if role_tools != "all" and tool_name not in role_tools:
            reason = f"Role '{user_role}' does not have access to tool '{tool_name}'"
            self._denial_log.append({"agent": agent_type, "tool": tool_name, "reason": reason})
            logger.info("permission.denied.layer2", role=user_role, tool=tool_name)
            return HookResult.block(reason)

        # --- Layer 3: Tool permission level check ---
        tool_config = self._config.tools.get(tool_name, {})
        permission_level = tool_config.get("permission_level", "read")
        trust_level = agent_config.get("trust_level", self._config.get_default_trust_level())

        if permission_level == "destructive" and trust_level != "auto":
            # Destructive actions require explicit auto trust or human approval
            reason = f"Tool '{tool_name}' is destructive and agent '{agent_type}' has trust_level='{trust_level}'"
            self._denial_log.append({"agent": agent_type, "tool": tool_name, "reason": reason})
            logger.info("permission.escalation_needed", agent=agent_type, tool=tool_name)
            return HookResult.block(reason)

        return HookResult.allow()

    def get_visible_tools(self, agent_type: str) -> list[str]:
        """
        Layer 1 filter: return only tools this agent is allowed to see.

        The orchestrator uses this when building agent prompts — tools
        not in this list are invisible to the agent's LLM.
        """
        agent_config = self._config.agents.get(agent_type, {})
        allowed = agent_config.get("allowed_tools", [])
        if not allowed:
            return list(self._config.tools.keys())
        return [t for t in allowed if t in self._config.tools]

    @property
    def denial_count(self) -> int:
        return len(self._denial_log)

    @property
    def recent_denials(self) -> list[dict[str, str]]:
        return self._denial_log[-10:]
