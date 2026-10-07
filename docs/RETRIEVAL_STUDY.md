# SWE-bench Verified localization study

## Verified baseline

The archived output from [GitHub Actions run 37561575185](https://github.com/RyanBalech/llm-agent-evaluation/actions/runs/37561575185)
at commit `b422b8321ca4b8fe93299840b63c8b0a8bf94ce9` contains 20 completed tasks and
zero failures. Per-task scores are committed in
`results/published/swebench_verified_bm25_20.json`.

| Metric | BM25 |
|---|---:|
| Mean file Recall@5 | 0.425000 |
| Mean reciprocal rank | 0.279167 |
| Mean nDCG@5 | 0.297423 |
| Mean retrieved characters | 39,428.6 |
| Mean retrieval latency (ms) | 288.1 |

Recall@5 is the fraction of gold files retrieved in the first five distinct files,
averaged across tasks. It is not simply a binary hit rate. These scores measure
file localization, not successful code patches or resolved SWE-bench instances.
Latency excludes repository checkout, chunking and index/embedding construction.

## Protocol

The materializer orders Verified test instances by SHA-256 of their instance ID,
then selects 20, excluding the three declared engineering smoke tasks. Each task
checks out its own base commit. The reference patch is used only to derive scoring
labels. The archived result contains exact task IDs but does not retain the dataset
revision or full source manifest, so future dataset changes must be checked against
those IDs before calling a rerun identical.

All methods use top-k 5, 50 candidate chunks, a 40,000-character budget, 80-line
chunks and 20-line overlap. CodeBERT dense retrieval and hybrid RRF runs remain
pending; no dense/hybrid gain is claimed here.

The comparison command rejects failed/incomplete studies, duplicate task IDs,
different ordered task sets and unequal context/chunk budgets. This prevents a
silent comparison on whichever easier tasks happened to finish. A 95% paired
bootstrap interval over tasks accompanies each method difference. The 20-task
slice is small, and tasks from the same repository may be correlated.

## Reproduce

```bash
pip install -e ".[dev,bench]"
python scripts/materialize_verified_slice.py --output .benchmark/study.jsonl --limit 20 --exclude-smoke
llm-agent-eval swebench --tasks .benchmark/study.jsonl --output results/bm25.json --method bm25 --top-k 5 --candidate-chunks 50 --context-budget-chars 40000 --lines-per-chunk 80 --overlap 20
```

The CLI now saves an atomic checkpoint after each task and prints progress.
Completed task checkouts are removed to bound disk use. Checkpoints preserve
partial evidence but do not implement automatic resume and must not be reported
as a complete benchmark. Dense/hybrid comparisons require the optional `ml`
dependencies and are available in the retrieval-comparison workflow.
