# Release verification

The application was verified locally on Windows with Python 3.13, FastAPI 0.143.0,
PyTorch 2.10.0+cu130, Transformers 4.57.6, and an NVIDIA RTX 4050 Laptop GPU.
The live static deployment is at
[ryanbalech.github.io/llm-agent-evaluation](https://ryanbalech.github.io/llm-agent-evaluation/).

## Checks performed

- Python tests cover the original retrieval/evaluation modules and the application API:
  demo search, budget enforcement, persistence across restart, deletion, authentication,
  origin and host checks, unsafe repository/ref rejection, source provenance, and interrupted jobs.
- Ruff passes for application code, tests, and scripts. All 58 Python tests pass locally.
- Browser integration passes at desktop and mobile sizes: demo indexing, search, source
  inspection, JSON download, all twenty evaluation rows, and live public GitHub indexing.
- The real pinned CodeBERT encoder loads on CUDA. BM25, dense, and hybrid searches all complete
  on the public project snapshot. [Raw smoke record](../results/published/product_smoke.json).
- All twenty materialized task IDs, source commits, issue hashes, and patch hashes match
  `configs/verified_20_manifest.json` at the pinned dataset revision.
- Static browser integration passes against a local HTTP server: the shared Python BM25 engine
  runs in Pyodide, searches both prepared snapshots, exports actual results, and preserves mobile layout.
- The same browser smoke test passes against the live GitHub Pages URL: repository selection,
  search, source inspection, JSON export, evaluation, and mobile layout.
- The real local Qwen2.5-Coder-1.5B-Instruct model generated a cited cancellation explanation on
  CUDA. Source excerpts are attached from the retrieved snapshot. This checks the generation path,
  not diagnosis quality across real issues. [Recorded output](../results/published/local_explanation_smoke.json).
- The separate repository-balanced lexical run completed all 50 tasks across 12 repositories,
  with Recall@5 0.555. This is a different task sample, not an improvement over the original 20-task score.
- Pull request #4 remains a draft. Its five CI jobs pass, including tests, both browser builds,
  a dense-cache check, and a container build plus health/demo request.

## Functional example

At repository commit `2587c2633d78c67d353e8f2d98611c6cc8135ba5`, a query about
`select_under_character_budget` ranks `src/llm_agent_eval/evaluation.py` first with BM25 and
hybrid. Dense retrieval ranks `src/llm_agent_eval/__init__.py` first. This is one functional
example, not a controlled quality comparison. Its recorded timings are individual requests;
they are not latency percentiles or throughput measurements.

## Unverified or pending

- The complete CodeBERT/hybrid comparison on the twenty- and fifty-task slices remains pending.
  The frozen runner preserves a per-task checkpoint; no gain is claimed from partial results.
- Repository-aware uncertainty estimates await completed paired comparisons. Sustained load
  tests and developer usability measurements have not been performed.
- Local Docker execution could not be run because the Docker daemon is unavailable on this
  workstation. The CI container job builds the image and checks health and authenticated demo.
- The public Pages build accepts only prepared repositories. Arbitrary public GitHub indexing,
  dense/hybrid retrieval, and the Python local model require the local application.

Reproduce the checks with the commands in [the product guide](PRODUCT.md). Screenshot artifacts
come from a running browser using actual API results, not static fixture screens.
