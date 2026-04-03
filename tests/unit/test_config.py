"""Tests for the configuration system."""

from pathlib import Path

import pytest

from unstruck.config import Config, ConfigError


@pytest.fixture
def config():
    return Config()


def test_loads_all_config_files(config):
    assert config.project["name"] == "unstruck-engine"
    assert config.models["tiers"]["orchestrator"]["primary"] == "gpt-4o"
    assert "analyst" in config.agents
    assert "rag_ingest" in config.tools
    assert "admin" in config.permissions["roles"]
    assert config.budgets["tokens"]["total_budget"] == 50000
    assert config.guardrails["input"]["prompt_injection"]["enabled"] is True


def test_model_tier_access(config):
    tier = config.get_model_tier("orchestrator")
    assert tier["primary"] == "gpt-4o"
    assert "fallback" in tier
    assert tier["temperature"] == 0

    tier = config.get_model_tier("worker")
    assert tier["primary"] == "gpt-4o-mini"


def test_model_tier_unknown_raises(config):
    with pytest.raises(ConfigError, match="Unknown model tier"):
        config.get_model_tier("nonexistent")


def test_model_pricing(config):
    inp, out = config.get_model_pricing("gpt-4o")
    assert inp == 2.50
    assert out == 10.00

    inp, out = config.get_model_pricing("unknown-model")
    assert inp == 0.0
    assert out == 0.0


def test_agent_config(config):
    agent = config.get_agent_config("analyst")
    assert agent["model_tier"] == "worker"
    assert agent["trust_level"] == "auto"
    assert "prompt" in agent


def test_agent_unknown_raises(config):
    with pytest.raises(ConfigError, match="Unknown agent"):
        config.get_agent_config("nonexistent")


def test_tool_config(config):
    tool = config.get_tool_config("rag_ingest")
    assert tool["permission_level"] == "write"


def test_tool_unknown_raises(config):
    with pytest.raises(ConfigError, match="Unknown tool"):
        config.get_tool_config("nonexistent")


def test_role_access(config):
    admin = config.get_role("admin")
    assert admin["agents"] == "all"
    assert admin["max_cost_per_session"] == 10.00

    user = config.get_role("user")
    assert "analyst" in user["agents"]


def test_budget_access(config):
    budgets = config.get_token_budgets()
    assert budgets["total_budget"] == 50000
    assert budgets["per_agent_budget"] == 8000

    limits = config.get_pipeline_limits()
    assert limits["max_iterations"] == 5


def test_prompt_loading(config):
    prompt = config.load_prompt("orchestrator/strategize.md")
    assert "Strategist" in prompt
    assert "{agent_list}" in prompt


def test_prompt_caching(config):
    prompt1 = config.load_prompt("agents/analyst.md")
    prompt2 = config.load_prompt("agents/analyst.md")
    assert prompt1 is prompt2  # Same object — cached


def test_prompt_missing_raises(config):
    with pytest.raises(ConfigError, match="Prompt file not found"):
        config.load_prompt("nonexistent/prompt.md")


def test_plugin_discovery(config):
    plugins = config.discover_plugins()
    # plugins/ is empty in scaffold — should return empty dict
    assert isinstance(plugins, dict)


def test_data_dir(config):
    data_dir = config.data_dir
    assert isinstance(data_dir, Path)
    assert data_dir.exists()


def test_circuit_breaker(config):
    cb = config.get_circuit_breaker()
    assert cb["max_retries"] == 3
    assert cb["backoff_base"] == 2


def test_compression_config(config):
    comp = config.get_compression_config()
    assert comp["auto_trigger_pct"] == 0.80
    assert comp["circuit_breaker_fails"] == 3
