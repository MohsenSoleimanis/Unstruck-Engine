"""Tests for MCP built-in tools."""

import tempfile
from pathlib import Path

import pytest

from mas.tools.mcp_client import MCPToolClient, configure_sandbox


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def client(tmp_dir):
    configure_sandbox(tmp_dir)
    c = MCPToolClient()
    c.register_builtin_tools()
    return c


def test_list_tools(client):
    tools = client.list_tools()
    names = [t["name"] for t in tools]
    assert "fs_read" in names
    assert "fs_write" in names
    assert "fs_list" in names
    assert "http_get" in names
    assert "db_query" in names
    assert "json_read" in names
    assert "csv_read" in names
    assert len(tools) == 11


@pytest.mark.asyncio
async def test_fs_write_and_read(client, tmp_dir):
    result = await client.call_tool("fs_write", {"path": "test.txt", "content": "hello world"})
    assert result["written"] is True

    result = await client.call_tool("fs_read", {"path": "test.txt"})
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_fs_list(client, tmp_dir):
    (tmp_dir / "a.txt").write_text("a")
    (tmp_dir / "b.txt").write_text("b")
    (tmp_dir / "c.py").write_text("c")

    result = await client.call_tool("fs_list", {"directory": ".", "pattern": "*.txt"})
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_fs_info(client, tmp_dir):
    (tmp_dir / "test.txt").write_text("hello")
    result = await client.call_tool("fs_info", {"path": "test.txt"})
    assert result["is_file"] is True
    assert result["extension"] == ".txt"
    assert result["size_bytes"] == 5


@pytest.mark.asyncio
async def test_path_traversal_blocked(client):
    result = await client.call_tool("fs_read", {"path": "../../etc/passwd"})
    assert "error" in result


@pytest.mark.asyncio
async def test_json_write_and_read(client, tmp_dir):
    await client.call_tool("json_write", {"path": "data.json", "data": {"key": "value", "num": 42}})
    result = await client.call_tool("json_read", {"path": "data.json"})
    assert result["data"]["key"] == "value"
    assert result["data"]["num"] == 42


@pytest.mark.asyncio
async def test_csv_read(client, tmp_dir):
    (tmp_dir / "data.csv").write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
    result = await client.call_tool("csv_read", {"path": "data.csv"})
    assert result["headers"] == ["name", "age", "city"]
    assert result["total_rows"] == 2
    assert result["rows"][0] == ["Alice", "30", "NYC"]


@pytest.mark.asyncio
async def test_db_create_and_query(client, tmp_dir):
    db_path = str(tmp_dir / "test.db")

    await client.call_tool("db_execute", {
        "database": db_path,
        "sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
    })

    await client.call_tool("db_execute", {
        "database": db_path,
        "sql": "INSERT INTO users (name, age) VALUES (?, ?)",
        "params": ["Alice", 30],
    })

    result = await client.call_tool("db_query", {
        "database": db_path,
        "sql": "SELECT * FROM users ORDER BY name",
    })
    assert result["count"] == 1
    assert result["rows"][0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_sql_injection_blocked(client, tmp_dir):
    db_path = str(tmp_dir / "test.db")

    # DROP TABLE should be blocked
    result = await client.call_tool("db_query", {
        "database": db_path,
        "sql": "DROP TABLE users",
    })
    assert "error" in result

    result = await client.call_tool("db_execute", {
        "database": db_path,
        "sql": "DROP TABLE users",
    })
    assert "error" in result


@pytest.mark.asyncio
async def test_unknown_tool(client):
    with pytest.raises(ValueError, match="not found"):
        await client.call_tool("nonexistent_tool", {})
