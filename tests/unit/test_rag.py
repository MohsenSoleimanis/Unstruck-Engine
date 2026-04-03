"""Tests for RAG service — structure and fallback parser."""

import pytest
from pathlib import Path

from unstruck.config import Config
from unstruck.rag.service import RAGService, _AVAILABLE
from unstruck.rag.tools import register_rag_tools
from unstruck.tools.registry import ToolRegistry


class TestRAGServiceStructure:
    def test_creates_with_config(self):
        config = Config()
        service = RAGService(config)
        assert service._initialized is False

    def test_availability_flag(self):
        # Should be True if raganything is installed
        assert _AVAILABLE is True

    def test_is_available_before_init(self):
        config = Config()
        service = RAGService(config)
        assert service.is_available is False  # Not initialized yet


class TestPyMuPDFFallbackParser:
    def test_parse_text_file(self, tmp_path):
        config = Config()
        service = RAGService(config)

        # Create a test text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world\n\nThis is a test document.", encoding="utf-8")

        content_list = service._parse_with_pymupdf(str(test_file))
        assert len(content_list) == 1
        assert content_list[0]["type"] == "text"
        assert content_list[0]["text"] == "Hello world\n\nThis is a test document."  # KEY: "text" not "content"

    def test_parse_nonexistent_file(self):
        config = Config()
        service = RAGService(config)
        content_list = service._parse_with_pymupdf("/nonexistent/file.pdf")
        assert content_list == []

    def test_content_format_uses_text_key(self, tmp_path):
        """Verify we use 'text' key, not 'content' — this was the v1 bug."""
        config = Config()
        service = RAGService(config)

        test_file = tmp_path / "test.txt"
        test_file.write_text("sample text")

        content_list = service._parse_with_pymupdf(str(test_file))
        item = content_list[0]
        assert "text" in item  # Correct key
        assert "content" not in item  # Old bug would have this


class TestRAGToolRegistration:
    def test_registers_two_tools(self):
        config = Config()
        service = RAGService(config)
        registry = ToolRegistry()

        register_rag_tools(registry, service)
        assert registry.has("rag_ingest")
        assert registry.has("rag_query")
        assert registry.count >= 2

    def test_tool_list_descriptions(self):
        config = Config()
        service = RAGService(config)
        registry = ToolRegistry()

        register_rag_tools(registry, service)
        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        assert "rag_ingest" in names
        assert "rag_query" in names

    @pytest.mark.asyncio
    async def test_rag_ingest_requires_file_path(self):
        config = Config()
        service = RAGService(config)
        registry = ToolRegistry()

        register_rag_tools(registry, service)
        result = await registry.call("rag_ingest")
        assert "error" in result
        assert "file_path" in result["error"]

    @pytest.mark.asyncio
    async def test_rag_query_requires_query(self):
        config = Config()
        service = RAGService(config)
        registry = ToolRegistry()

        register_rag_tools(registry, service)
        result = await registry.call("rag_query")
        assert "error" in result
        assert "query" in result["error"]
