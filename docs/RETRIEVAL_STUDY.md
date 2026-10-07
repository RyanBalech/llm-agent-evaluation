# SWE-bench Verified localization study

## Verified baseline

The archived output from [GitHub Actions run 37561575185](https://github.com/RyanBalech/llm-agent-evaluation/actions/runs/37561575185)
at commit `b422b8321ca4b8fe93299840b63c8b0a8bf94ce9` contains 20 completed tasks and
zero failures. Per-task scores are committed in
`results/published/swebench_verified_bm25_20.json`.

| Metric | BM25 + path prior |
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

The lexical baseline includes a filename/path-match weight of 0.15.

All methods use top-k 5, 50 candidate chunks, a 40,000-character budget, 80-line
chunks and 20-line overlap. The CodeBERT dense/hybrid comparison did not complete; no gain is claimed.
A real-encoder smoke task validates inference and cache reuse only, and is archived
separately in `results/published/dense_smoke.json`.

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

## Label-checked failure analysis

The archived Recall@5 and nDCG@5 are recomputed against recovered source records
before generating `results/published/bm25_error_analysis.json`. All 20 scores agree.
The audit manifest records the current dataset revision and hashes, and explicitly
does not claim that the original run archived that revision.

Eight tasks recover every gold file, one recovers half, and eleven recover none at
five. Twelve tasks are Django, three Astropy, two SymPy, two Sphinx and one sklearn.
Per-repository means on such small groups are descriptive, not reliable rankings.

Examples, with full file paths and base revisions in the artifact:

- `astropy__astropy-13453`: retrieves `astropy/io/ascii/tests/test_html.py` but misses
  the changed implementation `astropy/io/ascii/html.py`.
- `django__django-11740`: all five retrieved files are test `models.py` files; the
  changed file is `django/db/migrations/autodetector.py`.
- `astropy__astropy-8707`: retrieves `astropy/io/fits/header.py`, but misses the other
  changed file `astropy/io/fits/card.py`, giving Recall@5 = 0.5.

These traces motivate examining test/implementation confusions and repeated chunks
from the same file. They do not prove a semantic root cause or a fix's effectiveness.

New benchmark artifacts record input hashes, indexed file/chunk counts, unindexed
gold paths and setup time. Failures produce a nonzero CLI exit status after the
checkpoint is written. The encoder is loaded once, and dense/hybrid can share cached
embeddings keyed to source content and immutable weights. One shared hybrid class
now defines the candidate pool consistently across entry points.

## Paired sparse ablation

The follow-up rechecks all 20 base commits and scores seven arms. The original
40k-character baseline is reproduced exactly on every task for Recall@5, MRR and
nDCG@5. No task subset was dropped.

| Arm | Recall@5 | MRR | nDCG@5 | Mean code characters |
|---|---:|---:|---:|---:|
| Path prior, 5k cap | 0.200 | 0.167 | 0.175 | 4,258 |
| Path prior, 10k cap | 0.275 | 0.217 | 0.226 | 9,489 |
| Path prior, 20k cap | 0.375 | 0.258 | 0.276 | 19,400 |
| Path prior, 40k cap | 0.425 | 0.279 | 0.297 | 39,429 |
| Path prior, 80k cap | 0.425 | 0.288 | 0.297 | 79,446 |
| Plain BM25, 40k cap | 0.425 | 0.279 | 0.297 | 39,353 |
| Issue-independent file order, 40k cap | 0.000 | 0.000 | 0.000 | 39,742 |

The deterministic control sorts chunk IDs lexicographically and keeps the same
50-candidate pool and packing. It is a specific control, not an estimate of every
possible no-retrieval prompt. Plain BM25 still indexes paths as document text; only
the additional 0.15 path prior is removed.

The 5k-minus-40k Recall@5 difference is −0.225, with a task-bootstrap 95% interval
[−0.400, −0.050]. Larger context helps in that comparison, but there is no measured
top-five benefit from 40k to 80k or from the extra path prior. MRR increases slightly
at 80k because it is evaluated beyond the first five budgeted files.

These are exploratory follow-ups on the original slice; intervals do not correct
for multiple comparisons or same-repository dependence. Full scores and paired
intervals are in `results/published/sparse_ablation.json`.
