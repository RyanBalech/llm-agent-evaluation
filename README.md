# Repository-Level LLM Agent Evaluation

A file-localization benchmark for the retrieval stage of software-engineering agents.
Given an issue and its repository at the issue's base commit, rank candidate source
files under a fixed code-context budget. Reference patches supply scoring labels only.

## Measured baseline

On a fixed **20-task SWE-bench Verified slice**, the lexical baseline completed 20/20 tasks:

| Metric | BM25 + filename/path prior |
|---|---:|
| Mean file Recall@5 | 0.425 |
| Mean reciprocal rank | 0.279 |
| Mean nDCG@5 | 0.297 |

The existing baseline includes a path-match weight of 0.15; it is not unmodified
BM25. Gold-label verification finds eight complete top-five localizations, one
partial localization and eleven misses. Twelve of the twenty tasks are from Django.
These are localization scores, not resolved-issue or patch-correctness scores.

[Results and concrete failure cases](docs/RETRIEVAL_STUDY.md) ·
[Raw baseline](results/published/swebench_verified_bm25_20.json) ·
[Label-checked analysis](results/published/bm25_error_analysis.json) ·
[Design decisions](docs/METHOD.md)

The completed paired sparse ablation finds Recall@5 of 0.200, 0.275, 0.375,
0.425 and 0.425 at 5k, 10k, 20k, 40k and 80k code characters. Plain BM25 matches
the path-prior baseline on this slice; doubling 40k to 80k adds no top-five recall.
[All seven arms and per-task scores](results/published/sparse_ablation.json).

## Implemented methods

- BM25 with configurable filename/path prior and a no-issue file-order control.
- Transformer dense retrieval: attention-masked mean pooling, L2 normalization and cosine ranking.
- Sparse/dense reciprocal-rank fusion through one shared implementation.
- Distinct-file Recall@K, MRR and nDCG; whole-chunk packing under a character budget.
- Paired comparisons that reject mismatched budgets, task lists and incomplete runs.

Dense inference and cache reuse have been checked with the real CodeBERT encoder on
the engineering smoke task. The full dense/hybrid study has not completed; no gain is
claimed. The benchmark now loads one encoder per run, reuses content-addressed
embeddings, records resolved encoder revisions and writes an atomic checkpoint per task.

## Run

Python 3.10+. The basic lexical benchmark has no ML dependencies.

```bash
python -m pip install -e ".[dev,bench]"
python scripts/materialize_verified_slice.py --output .benchmark/study.jsonl --limit 20 --exclude-smoke
python -m llm_agent_eval.cli swebench --tasks .benchmark/study.jsonl \
  --output results/bm25.json --method bm25
python scripts/run_sparse_ablation.py .benchmark/study.jsonl
python scripts/analyze_localization.py results/bm25.json .benchmark/study.jsonl
python -m pytest -q
```

The materializer resolves a dataset branch to an immutable Hugging Face commit.
The frozen [20-task audit manifest](configs/verified_20_manifest.json) includes base
commits and issue/patch hashes. Dense/hybrid runs need `.[ml]`; the comparison workflow
pins CodeBERT weights and shares an embedding cache between methods.

For the measured local stack, see [requirements-reproduce.txt](requirements-reproduce.txt)
(Python 3.12; critical package pins).

## Code

- [benchmark.py](src/llm_agent_eval/benchmark.py): base-commit checkouts, manifest, progress and failures.
- [retrieval.py](src/llm_agent_eval/retrieval.py), [dense.py](src/llm_agent_eval/dense.py), [hybrid.py](src/llm_agent_eval/hybrid.py): ranking methods.
- [evaluation.py](src/llm_agent_eval/evaluation.py): context packing and file-level scoring.
- [comparison.py](src/llm_agent_eval/comparison.py): strict paired-artifact validation.

LLM reranking/provider interfaces are additional plumbing. Patch generation and an
iterative coding-agent loop are future work, not demonstrated capabilities.

## References

[SWE-bench retrieval guide](https://www.swebench.com/SWE-bench/guides/create_rag_datasets/)
and [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) provide benchmark and
agent context; this repository isolates retrieval instead of reproducing their agent scores.
