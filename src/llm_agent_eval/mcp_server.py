"""Local stdio MCP tools for issue localization over immutable RepoLens indexes."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from .repository import normalize_repository, validate_ref
from .service import CapacityError, IndexService


def create_server(service: IndexService) -> MCPServer:
    server = MCPServer("RepoLens", instructions=(
        "Index a workspace or public GitHub repository, wait until its status is ready, then "
        "search for source files relevant to an issue. Search scores are ranking signals; "
        "inspect cited source before drawing conclusions."
    ))

    def valid_id(index_id: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32}", index_id):
            raise ToolError("index_id must be a 32-character hexadecimal identifier")
        return index_id

    def tool_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise ToolError("Index or source file not found") from exc
        except (ValueError, CapacityError) as exc:
            raise ToolError(str(exc)) from exc

    def result(value: dict) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))],
                              structured_content=value)

    @server.tool()
    def index_workspace() -> CallToolResult:
        """Snapshot the configured local workspace, including current working-tree edits."""
        return result(tool_call(service.submit_workspace))

    @server.tool()
    def index_repository(repository: str, revision: str = "",
                         dense: bool = False) -> CallToolResult:
        """Start indexing a public GitHub repository at a branch, tag, or commit."""
        repository = tool_call(normalize_repository, repository)
        revision = tool_call(validate_ref, revision.strip())
        return result(tool_call(service.submit, repository, revision, dense=dense))

    @server.tool()
    def index_status(index_id: str) -> CallToolResult:
        """Check an indexing job; search only after its status becomes ready."""
        return result(tool_call(service.get, valid_id(index_id)))

    @server.tool()
    def list_indexes() -> CallToolResult:
        """List available source snapshots and their readiness status."""
        return result({"indexes": service.list()})

    @server.tool()
    def search_code(index_id: str, query: str, method: Literal["bm25", "dense", "hybrid"] = "bm25",
                    top_k: int = 5, context_budget_chars: int = 12000) -> CallToolResult:
        """Rank relevant files and return line-numbered source excerpts for an issue."""
        valid_id(index_id)
        query = query.strip()
        if not 3 <= len(query) <= 8000:
            raise ToolError("Query must contain 3 to 8,000 characters")
        if not 1 <= top_k <= 20 or not 1000 <= context_budget_chars <= 80000:
            raise ToolError("Use top_k 1–20 and context_budget_chars 1,000–80,000")
        return result(tool_call(service.search, index_id, query, method,
                                context_budget_chars, top_k))

    @server.tool()
    def read_indexed_file(index_id: str, path: str, start_line: int = 1,
                          max_lines: int = 120) -> CallToolResult:
        """Inspect a bounded line range from a source snapshot returned by search."""
        valid_id(index_id)
        if not 1 <= start_line <= 1_000_000 or not 1 <= max_lines <= 200:
            raise ToolError("Use start_line >= 1 and max_lines 1–200")
        return result(tool_call(service.read_file, index_id, path, start_line, max_lines))

    @server.tool()
    def remove_index(index_id: str) -> CallToolResult:
        """Remove a RepoLens source snapshot to free one index slot."""
        tool_call(service.delete, valid_id(index_id))
        return result({"removed_index_id": index_id})

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RepoLens MCP server over stdio")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="Local directory exposed by index_workspace (default: launch directory)")
    parser.add_argument("--data-dir", type=Path,
                        help="Private index directory (default: .repolens-mcp in the workspace)")
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    if not workspace.is_dir():
        parser.error("--workspace must name a directory")
    data_dir = args.data_dir or workspace / ".repolens-mcp"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = IndexService(data_dir, enable_dense=os.getenv("REPOLENS_ENABLE_DENSE") == "1",
                           workspace_root=workspace)
    try:
        create_server(service).run(transport="stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
