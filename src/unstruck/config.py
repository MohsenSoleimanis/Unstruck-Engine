"""Configuration system — loads YAML files, validates, provides typed access.

This is the foundation. Every other layer reads from here.
No hardcoded values in Python files — everything comes from config/.

Supports:
  - Layered configs: default.yaml → environment override → env vars
  - Prompt loading: reads Markdown files from prompts/
  - Plugin discovery: scans plugins/ for agent.yaml files
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root — where config/, prompts/, plugins/ live
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


class Config:
    """
    Central configuration — single source of truth.

    Loads YAML files from config/ directory. Provides typed access
    to every setting. Immutable after load — no runtime mutation.
    """

    def __init__(self, config_dir: Path | None = None, env: str = "default") -> None:
        self._config_dir = config_dir or PROJECT_ROOT / "config"
        self._prompts_dir = PROJECT_ROOT / "prompts"
        self._plugins_dir = PROJECT_ROOT / "plugins"
        self._data: dict[str, Any] = {}
        self._prompts_cache: dict[str, str] = {}

        self._load(env)

    def _load(self, env: str) -> None:
        """Load all config files. Environment-specific overrides merge on top of defaults."""
        # Load each config file
        self._data["default"] = self._load_yaml("default.yaml")
        self._data["models"] = self._load_yaml("models.yaml")
        self._data["agents"] = self._load_yaml("agents.yaml")
        self._data["tools"] = self._load_yaml("tools.yaml")
        self._data["permissions"] = self._load_yaml("permissions.yaml")
        self._data["budgets"] = self._load_yaml("budgets.yaml")
        self._data["guardrails"] = self._load_yaml("guardrails.yaml")

        # Environment override (e.g., config/production.yaml)
        if env != "default":
            override_path = self._config_dir / f"{env}.yaml"
            if override_path.exists():
                override = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
                self._deep_merge(self._data["default"], override)

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        """Load a single YAML file. Returns empty dict if file doesn't exist."""
        path = self._config_dir / filename
        if not path.exists():
            return {}
        content = path.read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}

    def _deep_merge(self, base: dict, override: dict) -> None:
        """Recursively merge override into base (mutates base)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    # --- Typed accessors ---

    @property
    def project(self) -> dict[str, Any]:
        return self._data["default"].get("project", {})

    @property
    def server(self) -> dict[str, Any]:
        return self._data["default"].get("server", {})

    @property
    def rag(self) -> dict[str, Any]:
        return self._data["default"].get("rag", {})

    @property
    def models(self) -> dict[str, Any]:
        return self._data["models"]

    @property
    def agents(self) -> dict[str, dict[str, Any]]:
        return self._data["agents"].get("agents", {})

    @property
    def tools(self) -> dict[str, dict[str, Any]]:
        return self._data["tools"].get("tools", {})

    @property
    def permissions(self) -> dict[str, Any]:
        return self._data["permissions"]

    @property
    def budgets(self) -> dict[str, Any]:
        return self._data["budgets"]

    @property
    def guardrails(self) -> dict[str, Any]:
        return self._data["guardrails"]

    # --- Model tier access ---

    def get_model_tier(self, tier: str) -> dict[str, Any]:
        """Get model config for a tier (orchestrator, worker, cheap, vision, embedding)."""
        tiers = self._data["models"].get("tiers", {})
        if tier not in tiers:
            raise ConfigError(f"Unknown model tier: '{tier}'. Available: {list(tiers.keys())}")
        return tiers[tier]

    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get (input_price, output_price) per 1M tokens for a model."""
        pricing = self._data["models"].get("pricing", {})
        if model not in pricing:
            return (0.0, 0.0)
        prices = pricing[model]
        return (prices[0], prices[1])

    def get_circuit_breaker(self) -> dict[str, Any]:
        return self._data["models"].get("circuit_breaker", {})

    # --- Agent config access ---

    def get_agent_config(self, agent_type: str) -> dict[str, Any]:
        """Get config for a specific agent."""
        agents = self.agents
        if agent_type not in agents:
            raise ConfigError(f"Unknown agent: '{agent_type}'. Available: {list(agents.keys())}")
        return agents[agent_type]

    # --- Tool config access ---

    def get_tool_config(self, tool_name: str) -> dict[str, Any]:
        """Get config for a specific tool."""
        tools = self.tools
        if tool_name not in tools:
            raise ConfigError(f"Unknown tool: '{tool_name}'. Available: {list(tools.keys())}")
        return tools[tool_name]

    # --- Permission access ---

    def get_role(self, role: str) -> dict[str, Any]:
        """Get permissions for a role."""
        roles = self._data["permissions"].get("roles", {})
        if role not in roles:
            raise ConfigError(f"Unknown role: '{role}'. Available: {list(roles.keys())}")
        return roles[role]

    def get_default_trust_level(self) -> str:
        return self._data["permissions"].get("default_trust_level", "confirm")

    # --- Budget access ---

    def get_token_budgets(self) -> dict[str, int]:
        return self._data["budgets"].get("tokens", {})

    def get_pipeline_limits(self) -> dict[str, int]:
        return self._data["budgets"].get("pipeline", {})

    def get_compression_config(self) -> dict[str, Any]:
        return self._data["budgets"].get("compression", {})

    def get_rate_limits(self) -> dict[str, int]:
        return self._data["budgets"].get("rate_limiting", {})

    # --- Prompt loading ---

    def load_prompt(self, path: str) -> str:
        """
        Load a prompt template from prompts/ directory.

        Args:
            path: Relative path from prompts/ (e.g., "orchestrator/strategize.md")

        Returns:
            The prompt content as a string.
        """
        if path in self._prompts_cache:
            return self._prompts_cache[path]

        full_path = self._prompts_dir / path
        if not full_path.exists():
            raise ConfigError(f"Prompt file not found: {path} (looked in {full_path})")

        content = full_path.read_text(encoding="utf-8")
        self._prompts_cache[path] = content
        return content

    # --- Plugin discovery ---

    def discover_plugins(self) -> dict[str, dict[str, Any]]:
        """
        Scan plugins/ directory for agent.yaml files.

        Returns:
            Dict of plugin_name → plugin config.
        """
        plugins: dict[str, dict[str, Any]] = {}

        if not self._plugins_dir.exists():
            return plugins

        for plugin_dir in self._plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            agent_yaml = plugin_dir / "agent.yaml"
            if not agent_yaml.exists():
                continue

            plugin_config = yaml.safe_load(agent_yaml.read_text(encoding="utf-8")) or {}
            plugin_config["_plugin_dir"] = str(plugin_dir)
            plugins[plugin_config.get("name", plugin_dir.name)] = plugin_config

        return plugins

    # --- Data directory ---

    @property
    def data_dir(self) -> Path:
        path = Path(self.project.get("data_dir", "./data"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_level(self) -> str:
        return self.project.get("log_level", "INFO")


@lru_cache(maxsize=1)
def get_config(env: str = "default") -> Config:
    """Get the global config singleton."""
    return Config(env=env)
