"""Exercise the MCP protocol boundary and local source snapshots."""
import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters

from llm_agent_eval.mcp_server import create_server
from llm_agent_eval.service import IndexService


async def ready(client, index_id):
    for _ in range(100):
        result = await client.call_tool("index_status", {"index_id": index_id})
        row = result.structured_content
        if row["status"] in {"ready", "failed"}:
            assert row["status"] == "ready", row
            return row
        await asyncio.sleep(.02)
    pytest.fail("MCP index job did not finish")


def test_workspace_mcp_tools_snapshot_search_and_inspection(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "orders.py").write_text(
        "def cancel_order(order):\n    restore_inventory(order)\n", encoding="utf-8")
    (workspace / ".repolens-mcp").mkdir()
    (workspace / ".repolens-mcp" / "hidden.py").write_text("private data", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("outside secret", encoding="utf-8")
    try:
        (workspace / "linked.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pass
    service = IndexService(tmp_path / "indexes", workspace_root=workspace)

    async def exercise():
        async with Client(create_server(service)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert {"index_workspace", "index_repository", "index_status", "search_code",
                    "read_indexed_file", "list_indexes", "remove_index"} <= names
            submitted = await client.call_tool("index_workspace")
            assert not submitted.is_error, submitted
            assert submitted.structured_content is not None, submitted
            index_id = submitted.structured_content["id"]
            row = await ready(client, index_id)
            assert row["files"] == 1
            response = await client.call_tool("search_code", {
                "index_id": index_id, "query": "cancel order restore inventory",
                "context_budget_chars": 1000})
            result = response.structured_content
            assert result["files"][0]["path"] == "orders.py"
            assert result["files"][0]["snippets"][0]["source_url"] is None
            excerpt = await client.call_tool("read_indexed_file", {
                "index_id": index_id, "path": "orders.py", "max_lines": 2})
            assert "1: def cancel_order(order):" in excerpt.structured_content["text"]
            (workspace / "orders.py").write_text("def changed(): pass\n", encoding="utf-8")
            old_excerpt = await client.call_tool("read_indexed_file", {
                "index_id": index_id, "path": "orders.py"})
            assert "cancel_order" in old_excerpt.structured_content["text"]
            assert "changed" not in old_excerpt.structured_content["text"]
            invalid = await client.call_tool("read_indexed_file", {
                "index_id": index_id, "path": "../outside.py"})
            assert invalid.is_error
            assert len((await client.call_tool("list_indexes")).structured_content["indexes"]) == 1
            removed = await client.call_tool("remove_index", {"index_id": index_id})
            assert removed.structured_content["removed_index_id"] == index_id

    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_mcp_rejects_invalid_inputs_and_pins_public_source(tmp_path, monkeypatch):
    from llm_agent_eval.indexing import CodeChunk

    def fetch(repository, revision, root):
        assert (repository, revision) == ("org/project", "main")
        return [CodeChunk("stock.py:1-1", "stock.py", 1, 1,
                          "def restore_inventory(): pass")], "a" * 40, {
            "source_bytes": 29, "skipped_files": 0}

    monkeypatch.setattr("llm_agent_eval.service.fetch_sources", fetch)
    service = IndexService(tmp_path / "indexes", workspace_root=Path(tmp_path))

    async def exercise():
        async with Client(create_server(service)) as client:
            bad_repo = await client.call_tool("index_repository", {
                "repository": "file:///etc/passwd"})
            assert bad_repo.is_error
            bad_id = await client.call_tool("search_code", {
                "index_id": "../outside", "query": "inventory"})
            assert bad_id.is_error
            submitted = await client.call_tool("index_repository", {
                "repository": "https://github.com/org/project", "revision": "main"})
            index_id = submitted.structured_content["id"]
            await ready(client, index_id)
            excerpt = await client.call_tool("read_indexed_file", {
                "index_id": index_id, "path": "stock.py"})
            assert excerpt.structured_content["source_url"] == (
                "https://github.com/org/project/blob/" + "a" * 40 + "/stock.py#L1-L1")

    try:
        asyncio.run(exercise())
    finally:
        service.close()


def test_stdio_entrypoint_accepts_a_real_client(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stock.py").write_text("def restore_inventory(): pass\n", encoding="utf-8")
    (workspace / "prices.py").write_text("def apply_discount(): pass\n", encoding="utf-8")
    launch = StdioServerParameters(
        command=sys.executable,
        args=["-m", "llm_agent_eval.mcp_server", "--workspace", str(workspace),
              "--data-dir", str(tmp_path / "mcp-data")],
    )

    async def exercise():
        async with Client(launch) as client:
            submitted = await client.call_tool("index_workspace")
            assert not submitted.is_error, submitted
            index_id = submitted.structured_content["id"]
            await ready(client, index_id)
            found = await client.call_tool("search_code", {
                "index_id": index_id, "query": "restore_inventory"})
            assert found.structured_content["files"][0]["path"] == "stock.py"

    asyncio.run(exercise())
