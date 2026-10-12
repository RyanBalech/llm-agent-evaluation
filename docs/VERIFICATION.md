# Release verification

The application was verified locally on Windows with Python 3.13, FastAPI 0.143.0,
PyTorch 2.10.0+cu130, Transformers 4.57.6, and an NVIDIA RTX 4050 Laptop GPU.

## Checks performed

- Python tests cover the original retrieval/evaluation modules and the application API:
  demo search, budget enforcement, persistence across restart, deletion, authentication,
  origin and host checks, unsafe repository/ref rejection, source provenance, and interrupted jobs.
- Ruff passes for application code, tests, and the new comparison/browser scripts.
- Browser integration passes at desktop and mobile sizes: demo indexing, search, source
  inspection, JSON download, all twenty evaluation rows, and live public GitHub indexing.
- The real pinned CodeBERT encoder loads on CUDA. BM25, dense, and hybrid searches all complete
  on the public project snapshot. [Raw smoke record](../results/published/product_smoke.json).
- All twenty materialized task IDs, source commits, issue hashes, and patch hashes match
  `configs/verified_20_manifest.json` at the pinned dataset revision.

## Functional example

At repository commit `2587c2633d78c67d353e8f2d98611c6cc8135ba5`, a query about
`select_under_character_budget` ranks `src/llm_agent_eval/evaluation.py` first with BM25 and
hybrid. Dense retrieval ranks `src/llm_agent_eval/__init__.py` first. This is one functional
example, not a controlled quality comparison. Its recorded timings are individual requests;
they are not latency percentiles or throughput measurements.

## Unverified or pending

- The full twenty-task CodeBERT/hybrid comparison has not been completed. The frozen runner
  and paired checks are available, but no new accuracy improvement is claimed.
- A larger held-out study, repository-aware uncertainty estimates, indexing memory profiles,
  sustained load tests, and developer usability measurements remain future work.
- Docker and CI definitions are included. Local container execution requires a running Docker
  daemon. Hosted deployment and a public service URL have not been provisioned.

Reproduce the checks with the commands in [the product guide](PRODUCT.md). Screenshot artifacts
come from a running browser using actual API results, not static fixture screens.
