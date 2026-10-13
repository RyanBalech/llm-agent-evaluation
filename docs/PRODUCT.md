# RepoLens product guide

## The workflow

1. Enter a public GitHub `owner/repo` or HTTPS repository URL. Optionally specify a branch,
   tag, or commit. RepoLens resolves it to a commit and indexes supported UTF-8 source blobs.
2. Wait for the index to become ready. Jobs and completed snapshots survive server restarts.
   An interrupted job is marked failed; resubmit it. Indexing never executes repository code.
3. Describe an issue. Function names and exact errors help lexical search. Select a method
   available for that index and a character budget, then search.
4. Inspect the first five distinct files and expand their source excerpts. Links open the exact
   indexed commit. Highlighted terms explain lexical overlap; they do not prove causation.
5. Export JSON to preserve the query, commit, configuration, selected code, scores, timings,
   and index fingerprint. Repeat with a different budget or method to investigate behavior.
6. Open Evaluation lab to inspect the archived 20-task study, including every miss.

The sample shop contains a deliberately incomplete cancellation function. It is a demonstration,
not a held-out evaluation. No account, model download, or internet connection is needed for it.

## API

Interactive OpenAPI documentation is at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Process health and version |
| `GET /api/config` | Server capabilities |
| `POST /api/indexes` | Queue `{repository, revision, dense}`; returns 202 and an index ID |
| `GET /api/indexes` | List persistent jobs and indexes |
| `GET /api/indexes/{id}` | Poll queued/indexing/ready/failed state |
| `DELETE /api/indexes/{id}` | Remove an inactive index and its source snapshot |
| `POST /api/demo` | Create or reuse the bundled shop index |
| `POST /api/search` | Search `{index_id, query, method, context_budget_chars, top_k}` |
| `GET /api/evaluation` | Archived baseline plus explicit limitations |
| `POST /api/explanations` | Queue local generation for the same request schema as search |
| `GET /api/explanations/{id}` | Poll generation state and retrieve validated citations |

Search defaults match the benchmark: 80-line chunks with 20-line overlap, 50 candidates,
40,000 characters, and five files. The product omits zero-score lexical candidates to avoid
presenting arbitrary files as matches. The archived evaluation keeps its original semantics.
The budget applies before file deduplication; overlapping code and files beyond the displayed
top five still count. It is a character budget, not a model token budget.

## Local operation

```powershell
python -m pip install -e ".[app,dev]"
repolens --port 8000
```

Data defaults to `.repolens/` (Git-ignored). Configure `REPOLENS_DATA_DIR` to move it.
SQLite stores job metadata; JSON stores immutable chunks. Embedding caches stay in
`embeddings/` under the data directory. Removing an index does not garbage-collect shared
embedding caches. Use one process; multiple Uvicorn workers are not supported.

Optional dense search:

```powershell
python -m pip install -e ".[app,ml]"
$env:REPOLENS_ENABLE_DENSE="1"
repolens
```

The first dense job downloads CodeBERT at a pinned revision. It uses CUDA when available,
otherwise CPU. Dense indexing and queries share an inference lock to limit concurrent GPU
use. Model inputs truncate at 512 tokens. CPU inference can be substantially slower.

## Container and network deployment

The included container installs the lexical application. Start Docker, then:

```powershell
$env:REPOLENS_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose up --build -d
```

Use **Server access** in the UI to enter this token. It stays in tab memory and is sent in the
Authorization header. Reloading requires entering it again. All `/api/*` endpoints require the
token when configured. The command refuses a non-loopback bind without `REPOLENS_TOKEN`.

Compose binds to localhost, uses a persistent data volume, drops Linux capabilities, runs as
a non-root user with a read-only root filesystem, and caps CPU, memory, and processes.
For a hosted demo, put a TLS reverse proxy in front, set `REPOLENS_HOSTS` to the actual
hostname, and configure the proxy to preserve Host. Share the token only with authorized users.
This is single-user authentication, not multi-tenant authorization. Do not advertise it as an
anonymous public indexing service. The static browser deployment below avoids a public indexing server.

Health: `/healthz`. Logs include indexing completion, failures, method, and search latency;
search queries and source code are not included in application log events. No uptime,
multi-user load, or production SLA has been measured.

## Operational boundaries

- Public GitHub only; no private credentials, arbitrary URLs, or local filesystem paths.
- Three pending jobs, twenty stored records, three in-memory retrieval indexes.
- At most 4,000 eligible source files, 20 MB eligible source, 500 KB per file, 20,000 chunks.
- Git commands have timeouts. Symlinks, submodules, binary and invalid UTF-8 blobs are skipped.
- Source limits are checked after fetch. The Git pack transfer has no hard byte quota;
  enforce disk quotas and isolated workers before accepting anonymous public workloads.
- Queued jobs run in one in-process worker. Job metadata survives a crash, but jobs do not
  automatically resume. An external queue would be the next step for multiple instances.
- The API does not run a coding agent, execute tests from indexed repositories, or generate patches.

## Validation and experiments

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m pip install playwright
python -m playwright install chromium
# With the app running in another terminal:
python scripts/smoke_ui.py --live
```

The browser test covers demo search, export, evaluation, mobile layout, and optionally live
GitHub indexing. Screenshots and exported results are written to Git-ignored `artifacts/`.
CI also defines browser and container jobs.

For the complete frozen comparison:

```powershell
python -m pip install -e ".[bench,ml]"
python scripts/materialize_verified_slice.py --output .benchmark/study.jsonl --limit 20 --exclude-smoke --revision c104f840cc67f8b6eec6f759ebc8b2693d585d4a
python scripts/run_frozen_comparison.py .benchmark/study.jsonl
```

The runner verifies all task IDs, base commits, issue hashes, and patch hashes against the
audited manifest before evaluating. It runs BM25, CodeBERT, then hybrid with the same settings,
shares the embedding cache, writes task checkpoints, and produces paired bootstrap intervals.
Completed matching methods can be reused; interrupted methods resume completed task checkpoints
only when inputs, configuration, and evaluation-code fingerprints match. This can take a long
time and fetch large repositories. Keep failures visible and do not report partial runs as complete.

## Local explanations

Install `.[app,ml]`, set `REPOLENS_ENABLE_EXPLANATIONS=1`, and restart the app. After searching,
click **Explain retrieved evidence**. This runs Qwen2.5-Coder-1.5B-Instruct locally at a pinned
model revision; there are no inference API calls or API charges. The initial model download is
approximately 3 GB. CPU inference requires several GB of RAM and can take over a minute;
set `REPOLENS_EXPLAIN_DEVICE=cuda` when a suitable GPU is available.

Generation receives the issue and bounded excerpts from up to three retrieved files. Every
finding must cite an existing evidence ID. The server attaches the original retrieved excerpt
and source metadata, ignoring model-supplied quotes, URLs, and line numbers. Invalid output becomes a
failed job rather than an unsupported explanation. Valid citations still do not establish that
the model's diagnosis is correct. No commands, tests, or patches are executed by this feature.

The model returns a suggested next check. Generated output includes model revision, token usage,
generation time, and source commit; it is included in JSON exports after generation completes.
Only one generation job runs at a time. Jobs expire when the server restarts.

## Hugging Face Spaces

`scripts/deploy_space.py` packages an explicit allowlist of application source and publishes a
Docker Space using the locally authenticated Hugging Face account. It requires a clean committed
working tree so deployment records an exact source revision. It requests no hardware upgrade.
CPU Basic has no hourly hardware charge, but creating a Docker Space now requires an eligible
paid account plan. See [Hugging Face's current hosting requirements](https://huggingface.co/docs/hub/spaces-overview).

```powershell
.venv\Scripts\hf.exe auth login
python scripts/deploy_space.py
```

Public-demo mode (`REPOLENS_PUBLIC_DEMO=1`) enables anonymous search of prepared repositories.
Arbitrary indexing and deletion require an operator token. The Space includes the sample shop
and the public project repository at the deployed revision. Local generation is limited to one
job at a time and twenty requests per hour across the demo. The operator can configure a token
with a Space secret; no account token is packaged into the application.

The Space's filesystem is ephemeral. Prepared indexes rebuild after a restart. Hugging Face
provides the public HTTPS endpoint and may suspend inactive CPU Basic Spaces. Runtime health and
browser behavior must be checked after each upload before calling the deployment live.

## Static browser deployment

GitHub Pages can serve an interactive demo without an application server. Build it with
`python scripts/build_pages.py`, then serve `dist/pages` with any static HTTP server.
With GitHub authentication configured, `python scripts/deploy_pages.py` publishes a clean
committed checkout to the `gh-pages` branch and configures Pages if it is not already enabled.
It refuses to overwrite an existing Pages configuration that points elsewhere.
The build publishes only the bundled example, tracked public Python files at the source commit,
the retrieval engine, and evaluation artifacts. It never exports local repository indexes.

Search uses the same Python BM25 implementation and context-budget selector inside a Pyodide
WebAssembly worker. Queries stay in the browser. Prepared source snapshots have immutable GitHub
links; arbitrary repository ingestion and dense retrieval remain available in the Python app.
The first search downloads the Python runtime from a version-pinned CDN.

On WebGPU-capable browsers, optional explanations run a quantized Qwen2.5-Coder-1.5B-Instruct
model through WebLLM. Clicking the explanation button starts an approximately 1 GB model download;
subsequent uses can reuse the browser cache. This is a different inference runtime from the Python
model, and its output is not assumed identical. Generated JSON is constrained to known evidence IDs,
then validated with the shared Python citation validator. The source snippets are copied from the
snapshot, never from generated code. No inference API, hosted GPU, or secret key is required.
Browser GPU support, available memory, network downloads, and cache eviction affect availability.

## Broader evaluation

The separate 50-task manifest samples all twelve available repositories with at most five
issues each, excludes the original twenty tasks and all smoke instances, and pins the dataset.
Materialize it with `python scripts/materialize_balanced_slice.py`, then run:

```powershell
python scripts/run_frozen_comparison.py .benchmark/balanced-50.jsonl --manifest configs/verified_balanced_50_manifest.json --output-dir results/balanced-50
```
