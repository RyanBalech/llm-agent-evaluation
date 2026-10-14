# Use RepoLens from an MCP client

RepoLens exposes its retrieval service through a local stdio MCP server. A coding tool can
index its workspace or a public GitHub repository, search for code related to an issue, and
inspect line-numbered excerpts. The server only reads source and writes its own index data;
it does not edit or execute repository code.

## Install and connect

Python 3.10+ and Git are required. In the RepoLens checkout:

```bash
python -m pip install -e ".[mcp]"
```

Configure your MCP client to launch RepoLens as a local process. Replace both absolute paths
with paths on your machine. Set `--workspace` to the project the coding tool is investigating:

```json
{
  "mcpServers": {
    "repolens": {
      "command": "/absolute/path/to/repolens/.venv/bin/python",
      "args": [
        "-m", "llm_agent_eval.mcp_server",
        "--workspace", "/absolute/path/to/project"
      ]
    }
  }
}
```

On Windows, use the Python executable under `.venv\Scripts\python.exe` and Windows
absolute paths. The executable must be from the environment where `.[mcp]` is installed.
To check the launch command before adding it to a client, run:

```bash
python -m llm_agent_eval.mcp_server --workspace /absolute/path/to/project
```

It stays open without printing a prompt because the MCP client communicates over standard
input and output. Stop it with Ctrl+C. Logs go to standard error.

## Tool workflow

1. Call `index_workspace` for the configured local directory, or `index_repository` with a
   public `owner/repository` and optional branch, tag, or commit. Both return an `index_id`.
2. Call `index_status` until the job is `ready`. `list_indexes` shows saved snapshots.
3. Call `search_code` with the `index_id` and a concrete issue description. Include an exact
   symbol name when known; BM25 treats `restore_inventory` as one token. The default is BM25,
   five distinct files, and a 12,000-character code budget.
4. Call `read_indexed_file` with a returned path to inspect up to 200 lines from the same
   snapshot. `remove_index` frees a saved index slot when it is no longer needed.

`index_workspace` captures current working-tree content, including uncommitted edits. Run it
again after editing files; earlier index IDs keep their original source snapshots. Workspace
results use a content hash instead of a Git commit. Public GitHub results include links pinned
to the fetched commit. Search scores are ranking signals, not evidence that a bug has been
found or fixed.

The local server stores indexes in `.repolens-mcp` inside the configured workspace by default;
use `--data-dir` to choose another location. It has the same source size and queue limits as
the application. Use one server process per data directory. Only public GitHub repositories
are accepted by `index_repository`; local private source can be indexed through the configured
workspace. The server is local stdio only and is not part of the GitHub Pages deployment.

Optional dense and hybrid search need `.[mcp,ml]` and `REPOLENS_ENABLE_DENSE=1` in the
server process. Set `dense=true` when indexing a public repository. BM25 is the recommended
default based on the [completed comparison](RETRIEVAL_STUDY.md).
