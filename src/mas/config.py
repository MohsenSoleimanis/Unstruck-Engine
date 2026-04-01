"""Central configuration for the Multi-Agent System."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    model_config = {"env_prefix": "MAS_"}

    orchestrator_model: str = "claude-sonnet-4-20250514"
    worker_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_retries: int = 3
    request_timeout: int = 120
    temperature: float = 0.0


class MemoryConfig(BaseSettings):
    model_config = {"env_prefix": "MAS_MEMORY_"}

    chroma_persist_dir: Path = Path("./data/chroma")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    local_cache_ttl: int = 3600
    shared_collection: str = "mas_shared"


class LLMOpsConfig(BaseSettings):
    model_config = {"env_prefix": "MAS_"}

    enable_tracing: bool = True
    enable_cost_tracking: bool = True
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ServerConfig(BaseSettings):
    model_config = {"env_prefix": "MAS_"}

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4


class MASConfig(BaseSettings):
    """Root configuration — aggregates all sub-configs."""

    model_config = {"env_prefix": "MAS_", "env_file": ".env", "env_file_encoding": "utf-8"}

    project_name: str = "multi-agent-system"
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    llmops: LLMOpsConfig = Field(default_factory=LLMOpsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> MASConfig:
    return MASConfig()
