# Unstruck Engine — Architecture Specification

## Purpose

Unstruck Engine is an **open multi-agent orchestration platform**. It sits between users and AI capabilities. You give it a task — it figures out which agents to use, what data to retrieve, how to reason over it, and delivers a grounded answer.

- Any agent can plug in
- Any data source can connect
- Any tool can be added
- The platform handles orchestration, context, memory, security, and quality
- You build capabilities by adding plugins, not by rebuilding the system

---

## Core Principle

**Everything is a plugin. The platform owns the flow.**

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                            UNSTRUCK ENGINE                                  │
│                    Multi-Agent Orchestration Platform                        │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                          AUTH & SECURITY GATE                               │
│                    (first thing, before anything)                            │
│                                                                             │
│  AUTHENTICATION              AUTHORIZATION              DATA SECURITY       │
│  ┌─────────────────┐        ┌─────────────────┐        ┌────────────────┐  │
│  │ API key          │        │ Roles:           │        │ Secrets:       │  │
│  │ OAuth 2.0        │        │  admin/user/     │        │  encrypted     │  │
│  │ JWT tokens       │        │  viewer/service  │        │  never logged  │  │
│  │ Service-to-      │        │                  │        │  never in LLM  │  │
│  │ service auth     │        │ Per-user:        │        │  rotatable     │  │
│  │                  │        │  agents, tools,  │        │                │  │
│  │ Who are you?     │        │  budget, data    │        │ Isolation:     │  │
│  │                  │        │                  │        │  user A ≠ B    │  │
│  │                  │        │ Per-agent:       │        │  per-user KG   │  │
│  │                  │        │  tools, trust,   │        │  per-user RAG  │  │
│  │                  │        │  data access     │        │                │  │
│  │                  │        │                  │        │ Transport:     │  │
│  │                  │        │ Per-tool:        │        │  HTTPS/WSS     │  │
│  │                  │        │  who, what args, │        │  MCP auth      │  │
│  │                  │        │  read/write      │        │                │  │
│  │                  │        │                  │        │ Sanitization:  │  │
│  │                  │        │ Per-session:     │        │  path traversal│  │
│  │                  │        │  rate limit,     │        │  SQL injection │  │
│  │                  │        │  cost ceiling,   │        │  Cypher inject │  │
│  │                  │        │  concurrent cap  │        │  prompt inject │  │
│  └─────────────────┘        └─────────────────┘        └────────────────┘  │
│                                                                             │
│  Request → Authenticate → Authorize → Rate limit → PASS or REJECT          │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                           HOOK SYSTEM                                       │
│             (extension points — fires at every boundary)                    │
│             (this is how all layers connect without coupling)                │
│                                                                             │
│  ★ SessionStart                                                             │
│  │  Auth checks user quota │ Memory loads session │ LLMOps starts trace     │
│  │  Enterprise injects custom rules │ Resources initialized                 │
│  │                                                                          │
│  ★ PreLLMCall                                                               │
│  │  Context Engine checks budget │ Guardrails detect prompt injection       │
│  │  Cost control blocks if over ceiling │ Audit logs the prompt             │
│  │  Enterprise compliance rules applied                                     │
│  │                                                                          │
│  ★ PostLLMCall                                                              │
│  │  Grounding extracts citations │ Guardrails detect PII in output          │
│  │  Memory caches response in Tier 2 │ Cost records tokens spent            │
│  │  Quality scoring runs │ Audit logs the response                          │
│  │                                                                          │
│  ★ PreToolUse                                                               │
│  │  Permission system checks: user + agent + tool allowed?                  │
│  │  Auth validates: user has access to this resource?                        │
│  │  Can BLOCK (error fed back to agent) │ Can MODIFY args                   │
│  │  Audit logs the attempt                                                  │
│  │                                                                          │
│  ★ PostToolUse                                                              │
│  │  Memory caches result in Tier 2 │ Audit logs result                      │
│  │  Output sanitized (strip secrets, PII) │ Shared memory updated           │
│  │                                                                          │
│  ★ PreCompress                                                              │
│  │  Memory saves critical info to Tier 3 before it's compressed away        │
│  │  Marks "do not compress" items                                           │
│  │                                                                          │
│  ★ SessionEnd                                                               │
│  │  Session saved │ Audit flushed │ Resources cleaned │ Metrics recorded    │
│  │  Notifications sent │ Downstream systems triggered                       │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                          CONTEXT ENGINE                                     │
│                 (every LLM call goes through here)                          │
│                                                                             │
│  ┌───────────────┐ ┌────────────────┐ ┌────────────┐ ┌──────────────┐     │
│  │   RETRIEVAL    │ │  COMPRESSION   │ │  ASSEMBLY  │ │  GROUNDING   │     │
│  │               │ │                │ │            │ │              │     │
│  │ Asks:         │ │ Micro:         │ │ Builds the │ │ Citations    │     │
│  │  RAG-Anything │ │  drop redundant│ │ final      │ │ Source refs  │     │
│  │  Web search   │ │  tool outputs  │ │ prompt:    │ │ Confidence   │     │
│  │  Memory Tier2 │ │  deduplicate   │ │            │ │ scores       │     │
│  │  Memory Tier3 │ │                │ │ System     │ │              │     │
│  │  Session      │ │ Auto:          │ │ + Retrieved│ │ Verify:      │     │
│  │  DB query     │ │  summarize at  │ │   context  │ │ claim exists │     │
│  │  Any strategy │ │  80% window    │ │ + History  │ │ in source    │     │
│  │               │ │  circuit break │ │   (compr.) │ │              │     │
│  │ Merges all    │ │  after 3 fails │ │ + Agent    │ │ Strip PII    │     │
│  │ Ranks by      │ │                │ │   instruct │ │ from output  │     │
│  │ relevance     │ │ Full:          │ │ + Token    │ │              │     │
│  │               │ │  reset with    │ │   budget   │ │              │     │
│  │               │ │  summary       │ │   enforced │ │              │     │
│  └───────────────┘ └────────────────┘ └────────────┘ └──────────────┘     │
│                                                                             │
│  Fires hooks: PreLLMCall (before) → PostLLMCall (after)                    │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                       ORCHESTRATOR (THE BRAIN)                              │
│                                                                             │
│  1. UNDERSTAND                                                              │
│     Parse intent │ Load session │ Classify task type                        │
│     Detect follow-up │ Check what's already known                           │
│                                                                             │
│  2. VALIDATE & GUARD                                                        │
│     Input safety │ Prompt injection detection │ PII scan                    │
│     Ethical check │ Scope check │ Compliance │ Access control                │
│     → REJECT if unsafe │ FLAG if borderline │ PROCEED if clean              │
│                                                                             │
│  3. STRATEGIZE                                                              │
│     Select agents │ Decompose tasks │ Build dependency graph                │
│     Allocate token budget per agent │ Plan fallbacks                        │
│     Model tiering: cheap for routing, expensive for reasoning               │
│     → Structured TaskLedger                                                 │
│                                                                             │
│  4. DELEGATE                                                                │
│     Route tasks to agents │ Parallel if no deps │ Monitor progress          │
│     Context Engine feeds each agent │ PipelineContext blackboard             │
│                                                                             │
│  5. EVALUATE                                                                │
│     Output validation │ Citation verification │ Quality scoring             │
│     Consistency across agents │ Completeness check │ PII strip              │
│     → Structured ProgressLedger (LLM reflection)                            │
│                                                                             │
│  6. DECIDE                                                                  │
│     Good? → Synthesize                                                      │
│     Bad? → Replan with errors-as-feedback                                   │
│     Unsure? → Human-in-the-loop (pause, ask user, resume)                  │
│     Budget out? → Best effort with disclaimer                               │
│     Provider down? → Circuit breaker → fallback model chain                 │
│                                                                             │
│  7. LEARN                                                                   │
│     Update Memory Tier 3 │ Save session │ Record audit log                  │
│     Update LLMOps metrics │ Checkpoint state for recovery                   │
│                                                                             │
│  Cross-cutting:                                                             │
│    Human-in-the-loop │ Errors-as-feedback │ Circuit breakers                │
│    Checkpointing │ Model tiering │ Structured ledgers                       │
│                                                                             │
│  LangGraph cycle:                                                           │
│     PLAN → EXECUTE → REVIEW → (REPLAN | SYNTHESIZE) → END                  │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                           AGENT LAYER                                       │
│                                                                             │
│  AGENT REGISTRY                                                             │
│    Agents register: name, description, version, input/output types,         │
│    allowed_tools (scoped), trust_level (auto-approve vs ask user)           │
│    Orchestrator queries registry → selects agents for task                  │
│                                                                             │
│  Built-in: analyst, synthesizer, kg_reasoner                                │
│  Plugin: protocol_extractor, web_researcher, code_executor, data_analyst    │
│                                                                             │
│  Every agent:                                                               │
│    → Goes through Context Engine for LLM calls                             │
│    → Goes through Permission System for tool calls                         │
│    → Goes through Hook System at every boundary                            │
│    → Can read/write Memory Layer                                           │
│    → Can call Tool Layer via MCP                                           │
│    → Errors returned as feedback, not exceptions                           │
│                                                                             │
│  PERMISSION SYSTEM                                                          │
│    Layer 1: Registry filter (invisible tools)                               │
│    Layer 2: Per-call check (args, patterns, PreToolUse hook)                │
│    Layer 3: Human escalation (pause, ask user, resume/deny)                 │
│    Denial tracking (don't re-ask, feed back as context)                     │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                        TOOL LAYER (MCP)                                     │
│              (every external action, sandboxed, logged)                     │
│                                                                             │
│  TOOL REGISTRY                                                              │
│    Tools register: name, description, input/output schema,                  │
│    permission_level, sandbox_rules                                          │
│                                                                             │
│  Built-in: filesystem, HTTP, database, JSON/CSV, rag_ingest                │
│  Plugin (MCP): web search, code exec, email, Slack, any MCP server         │
│                                                                             │
│  Call flow: Agent → PreToolUse hook → permission check → sandbox →          │
│             execute → PostToolUse hook → cache in Tier 2 → return           │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                          MEMORY LAYER                                       │
│                                                                             │
│  TIER 2: AGENT CACHE (fast, per-agent, TTL-based)                          │
│    LLM response cache │ Tool result cache │ Working memory                  │
│    ENFORCED: checked before every LLM/tool call via hooks                  │
│    Written by: PostLLMCall hook, PostToolUse hook                          │
│                                                                             │
│  TIER 3: PERSISTENT STORE (survives across sessions)                       │
│    Session store (pipeline context, ingested docs, message history)         │
│    Audit log (every action: who, what, when, why, cost)                    │
│    Checkpoint store (LangGraph SQLite, resume after crash)                  │
│    Optional backends: Neo4j, PostgreSQL, Redis, S3 (via MCP tools)         │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                            LLMOps                                           │
│                                                                             │
│  Cost tracking (per agent/task/session/user, CEILINGS that halt)           │
│  Tracing (full pipeline, replay, debug mode)                               │
│  Evaluation (completeness, grounding, consistency, quality score)           │
│  Monitoring (health, latency, memory, sessions, alerts)                    │
│  Model tiering (cheap for routing, expensive for reasoning)                │
│                                                                             │
╞═════════════════════════════════════════════════════════════════════════════╡
│                                                                             │
│                          API + UI LAYER                                     │
│                                                                             │
│  FastAPI: REST, SSE streaming, WebSocket, Swagger, auth middleware         │
│  React: sidebar (conversations, files), chat (streaming, human-in-loop),   │
│         right panel (agents, KG, sources, costs, permissions)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         ALL LAYERS CONNECT TO THIS SERVICE:

┌─────────────────────────────────────────────────────────────────────────────┐
│                       RAG-ANYTHING (SERVICE)                                │
│                                                                             │
│  One instance per user/org. Multiple callers. Not split.                    │
│                                                                             │
│  Docling parser │ LightRAG KG │ Modal processors │ LightRAG vectors        │
│                                                                             │
│  Tool Layer calls .ingest(file) — parse + build KG + index                 │
│  Context Engine calls .query(question) — find relevant context              │
│  KG Reasoner calls .query(question) — graph-based retrieval                 │
│                                                                             │
│  Data isolation: per-user working_dir, enforced by Auth layer              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Information Flow

```
User Message + (optional file)
  │
  ▼
Auth Gate: authenticate → authorize → rate limit
  │
  ▼
★ SessionStart hook fires
  │
  ▼
Context Engine: retrieval (RAG-Anything, memory, session) → compression → assembly
  │
  ▼
Orchestrator Brain:
  1. Understand (intent, session, task type, follow-up detection)
  2. Validate & Guard (safety, PII, compliance)
  3. Strategize (agents, tasks, budget, fallbacks) → TaskLedger
  4. Delegate (route to agents, parallel execution)
     │
     ├─► Agent calls LLM → ★PreLLMCall hook → Context Engine → LLM → ★PostLLMCall hook
     ├─► Agent calls tool → ★PreToolUse hook → Permission check → Sandbox → Execute → ★PostToolUse hook
     ├─► Agent reads memory → Tier 2 cache check → Tier 3 if miss
     └─► Agent writes memory → Tier 2 cache + Tier 3 persistent
     │
  5. Evaluate (quality, citations, consistency) → ProgressLedger
  6. Decide:
     ├─► Good → Synthesize
     ├─► Bad → Replan (errors-as-feedback)
     ├─► Unsure → Human-in-the-loop (pause, ask, resume)
     ├─► Budget out → Best effort with disclaimer
     └─► Provider down → Circuit breaker → fallback model
  7. Learn (memory, session, audit, metrics)
  │
  ▼
★ SessionEnd hook fires
  │
  ▼
LLMOps: cost recorded, trace saved, metrics updated, audit flushed
  │
  ▼
Response to user (streamed via SSE)
```

---

## What's Justified By Production Evidence

| Feature | Evidence |
|---|---|
| LangGraph orchestrator loop | Every production system (Claude Code, OpenAI, Google ADK) |
| Structured ledgers | Magentic-One (shipped, benchmarked) |
| Context compression (3-layer) | Claude Code's most engineered feature |
| Human-in-the-loop | 68% of surviving deployments use it (arXiv 2512.04123) |
| Permission system (3-layer) | Claude Code (per-tool, 6 modes, denial tracking) |
| Circuit breakers + fallbacks | LinkedIn built entire system on messaging infra for this |
| Cost ceilings that halt | Ramp production agents (financial transactions) |
| Errors-as-feedback | Claude Code core pattern (model adapts, not crash) |
| Hook system | Claude Code (6 hooks), extensibility without core changes |
| Model tiering | 40-60% cost reduction (LangChain production data) |
| Session persistence | Every production system |
| Audit log | 89% of production agent deployments have observability |

## What Was Removed (research-only, no production evidence)

| Removed | Why |
|---|---|
| A2A peer-to-peer messaging | No production system uses it. All use orchestrator-mediated dispatch |
| TEA Protocol | Zero production adoption |
| MAGMA Multi-Graph | Research paper only, zero deployments |
| Communication topologies (bus/ring/tree) | Every production system uses star (orchestrator). Nobody implements formal topologies |
| Our separate NetworkX KG | LightRAG IS the KG. No duplication |
| Our separate ChromaDB | LightRAG has its own vectors. No duplication |

---

## Configuration & Prompts (nothing hardcoded in Python)

### Principle

The platform code **never changes** when you add agents, tools, prompts, or change settings. Everything is external config.

### Directory Structure

```
MAP/
├── config/                          # All configuration — YAML, layered
│   ├── default.yaml                 # Base config (all defaults)
│   ├── models.yaml                  # LLM models, pricing, fallback chains
│   ├── agents.yaml                  # Agent registry definitions
│   ├── tools.yaml                   # Tool registry definitions
│   ├── permissions.yaml             # Permission rules
│   ├── budgets.yaml                 # Token/cost budgets
│   └── guardrails.yaml             # Safety rules (PII patterns, blocked terms)
│
├── prompts/                         # All prompts — Markdown, versioned
│   ├── orchestrator/                # Brain prompts
│   │   ├── understand.md            # Parse user intent
│   │   ├── strategize.md            # Create task plan
│   │   ├── evaluate.md              # Review quality
│   │   └── decide.md               # What to do next
│   │
│   ├── agents/                      # Agent system prompts
│   │   ├── analyst.md               # Analyst reasoning prompt
│   │   ├── synthesizer.md           # Synthesizer fusion prompt
│   │   ├── kg_reasoner.md           # KG reasoning prompt
│   │   └── {agent_name}.md          # Any new agent — just add a file
│   │
│   └── guardrails/                  # Safety prompts
│       ├── input_safety.md          # Prompt injection detection
│       ├── output_safety.md         # PII/hallucination check
│       └── ethical.md               # Ethical boundary check
│
├── plugins/                         # Plugin agents (self-contained)
│   ├── protocol_extractor/
│   │   ├── agent.yaml               # Registration config
│   │   ├── prompts/                 # Its own prompts
│   │   └── agent.py                 # Its logic
│   │
│   ├── web_researcher/
│   │   ├── agent.yaml
│   │   ├── prompts/
│   │   └── agent.py
│   │
│   └── {your_new_agent}/           # Add a folder = add an agent
│       ├── agent.yaml
│       ├── prompts/
│       └── agent.py
│
└── src/mas/                         # Platform code (NEVER changes for new agents/tools)
```

### Config Files

**`config/models.yaml`** — LLM models and fallback chains
```yaml
models:
  orchestrator:
    primary: gpt-4o
    fallback: [claude-sonnet-4-20250514, gpt-4o-mini]
    temperature: 0
    max_tokens: 4096

  worker:
    primary: gpt-4o-mini
    fallback: [gpt-4.1-nano]
    temperature: 0
    max_tokens: 8192

  vision:
    primary: gpt-4o
    fallback: [claude-sonnet-4-20250514]

  cheap:  # For routing, classification, review
    primary: gpt-4o-mini
    fallback: [gpt-4.1-nano]

pricing:  # Per 1M tokens (input/output)
  gpt-4o: [2.50, 10.00]
  gpt-4o-mini: [0.15, 0.60]
  claude-sonnet-4-20250514: [3.00, 15.00]
```

**`config/agents.yaml`** — Agent registry definitions
```yaml
agents:
  analyst:
    description: "Reasons over retrieved context, produces grounded answers"
    version: "1.0.0"
    model_tier: worker          # Uses worker model (cheap)
    allowed_tools: []           # No tools — pure reasoning
    trust_level: auto           # No human approval needed
    prompt: prompts/agents/analyst.md

  synthesizer:
    description: "Fuses outputs from multiple agents"
    model_tier: worker
    allowed_tools: []
    trust_level: auto
    prompt: prompts/agents/synthesizer.md

  kg_reasoner:
    description: "Graph traversal + LLM reasoning for multi-hop questions"
    model_tier: worker
    allowed_tools: [rag_query]
    trust_level: auto
    prompt: prompts/agents/kg_reasoner.md
```

**`config/tools.yaml`** — Tool registry
```yaml
tools:
  filesystem:
    read:
      description: "Read a file (sandboxed)"
      permission_level: read
      sandbox: {root: "./data"}
    write:
      description: "Write a file (sandboxed)"
      permission_level: write
      sandbox: {root: "./data"}

  http:
    get:
      description: "HTTP GET request"
      permission_level: read
      domain_whitelist: []  # Empty = all allowed

  database:
    query:
      description: "SQL SELECT query"
      permission_level: read
      allowed_prefixes: [SELECT, PRAGMA, EXPLAIN]
    execute:
      description: "SQL write operation"
      permission_level: write
      blocked_keywords: [DROP, TRUNCATE]

  rag_ingest:
    description: "Ingest document via RAG-Anything"
    permission_level: write
    trust_level: auto

  rag_query:
    description: "Query RAG-Anything knowledge graph"
    permission_level: read
    trust_level: auto
```

**`config/permissions.yaml`** — Who can do what
```yaml
roles:
  admin:
    agents: all
    tools: all
    max_cost_per_session: 10.00
    max_cost_per_day: 100.00

  user:
    agents: [analyst, synthesizer, kg_reasoner]
    tools: [filesystem.read, http.get, database.query, rag_ingest, rag_query]
    max_cost_per_session: 1.00
    max_cost_per_day: 10.00

  viewer:
    agents: [analyst]
    tools: [rag_query]
    max_cost_per_session: 0.10
    max_cost_per_day: 1.00

agent_trust_levels:
  auto: "Execute without asking user"
  confirm: "Show plan, ask user before executing"
  strict: "Ask user before every tool call"
```

**`config/budgets.yaml`** — Token and cost budgets
```yaml
defaults:
  total_budget_tokens: 50000
  per_agent_budget_tokens: 8000
  context_budget_tokens: 12000
  synthesis_threshold: 0.85    # Trigger early synthesis at 85% budget
  max_iterations: 5

compression:
  auto_trigger_pct: 0.80       # Auto-compress at 80% window
  circuit_breaker_fails: 3     # Stop trying after 3 compression failures
  full_reset_budget: 50000     # Working budget after full reset
```

**`config/guardrails.yaml`** — Safety rules
```yaml
input:
  prompt_injection:
    enabled: true
    patterns:
      - "ignore previous instructions"
      - "you are now"
      - "system prompt"
    action: reject  # reject | flag | log

  pii_detection:
    enabled: true
    types: [email, phone, ssn, credit_card]
    action: flag

output:
  pii_stripping:
    enabled: true
    types: [email, phone, ssn, credit_card]

  hallucination_check:
    enabled: true
    require_citations: true
    min_confidence: 0.5

  ethical:
    blocked_topics: []
    escalate_to_human: true
```

### Prompts

Every prompt is a Markdown file. Loaded at runtime. Changeable without code deploy.

**`prompts/orchestrator/strategize.md`** example:
```markdown
You are the Strategist in the Unstruck Engine orchestrator.

Given the user's request and available agents, create an execution plan.

## Available Agents
{agent_list}

## Rules
1. Each task targets one agent from the list above.
2. Use dependency indices [0, 1, 2...] for ordering.
3. Independent tasks have no dependencies (run in parallel).
4. For document questions: ingest → query → analyze.
5. Allocate token budgets per agent from the total: {total_budget}.

## User Request
{user_query}

## Session Context
{session_context}

## Output
JSON array of task objects.
```

**`prompts/agents/analyst.md`** example:
```markdown
You are an Analyst agent. Answer the question using ONLY the provided context.

## Rules
1. Every claim must cite a source (page number, chunk ID, or entity name).
2. If the context doesn't contain enough information, say so explicitly.
3. Rate your confidence: high (directly stated), medium (inferred), low (speculative).
4. Structure your answer clearly with key points.

## Context
{context}

## Question
{question}
```

### Plugin Agents

Each plugin is a self-contained folder. The platform auto-discovers and registers them.

**`plugins/protocol_extractor/agent.yaml`**:
```yaml
name: protocol_extractor
description: "Extracts clinical trial protocol data (endpoints, eligibility, design)"
version: "1.0.0"
model_tier: worker
allowed_tools: [rag_query, rag_ingest]
trust_level: auto
prompt: prompts/extract.md
```

**Adding a new agent**: 
1. Create folder in `plugins/`
2. Write `agent.yaml` (registration) + `prompts/` (its prompts) + `agent.py` (logic)
3. Restart server
4. Orchestrator discovers it, planner can use it

No platform code changes. Ever.

### What Changes Where

| I want to... | I change... | Code change? |
|---|---|---|
| Add a new agent | `plugins/{name}/` folder | No |
| Change a prompt | `prompts/{file}.md` | No |
| Change LLM model | `config/models.yaml` | No |
| Add a fallback model | `config/models.yaml` | No |
| Change permissions | `config/permissions.yaml` | No |
| Change token budget | `config/budgets.yaml` | No |
| Add a safety rule | `config/guardrails.yaml` | No |
| Add a new MCP tool | Connect MCP server + `config/tools.yaml` | No |
| Change agent's allowed tools | `config/agents.yaml` | No |
| Change trust levels | `config/permissions.yaml` | No |
| Add enterprise rules | Hook functions (Python) | Minimal |
| Add a new retrieval strategy | `src/mas/context/` | Yes (platform extension) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (Python) |
| LLM | OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude) via LangChain |
| RAG | RAG-Anything (Docling + LightRAG) |
| Tools | MCP Protocol |
| API | FastAPI |
| UI | React 18 + TypeScript + Tailwind CSS |
| State | Zustand (frontend), Session files (backend) |
| Checkpointing | LangGraph SQLite checkpointer |
| Token counting | tiktoken |
| Tracing | Langfuse / OpenTelemetry |
| Auth | JWT + OAuth 2.0 |
