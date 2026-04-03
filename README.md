# Unstruck Engine

Multi-agent orchestration platform — config-driven, plugin-based, production-grade.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

## Quick Start

```bash
pip install -e ".[dev]"
unstruck serve
```

## Project Structure

```
config/          # YAML configuration (models, agents, tools, permissions, budgets)
prompts/         # Markdown prompt templates (orchestrator, agents, guardrails)
plugins/         # Plugin agents (self-contained: yaml + prompts + code)
src/unstruck/    # Platform code (never changes for new agents/tools)
ui/              # React frontend
tests/           # Unit + integration tests
```
