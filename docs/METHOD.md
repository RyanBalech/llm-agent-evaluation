# Retrieval design and scoring

## What enters the retriever?

Only the problem statement and source files at the declared base commit. The gold
patch stays in the evaluation record and supplies changed-file labels after ranking.
Issue and patch SHA-256 hashes identify the input records. Gold paths and unindexed
gold paths are recorded as diagnostics, never passed to `search()`.

## Baseline details

Code is split into 80-line chunks with 20-line overlap. BM25 indexes path plus code;
its default additional path-match weight is 0.15. The sparse ablation distinguishes
this baseline from plain BM25 with path weight zero. Tokenization preserves underscore
identifiers rather than splitting camelCase or learning a code vocabulary.

The first 50 ranked chunks are greedily packed under the code-character budget.
An oversized chunk is skipped, and subsequent smaller chunks may fit. Characters
exclude prompts, file headers, separators and model tokenization; this is not an
exact LLM token budget. Overlapping chunks are counted in full, so duplicate code
can consume budget. Distinct files are ranked by their first selected chunk.

Recall@5 is the fraction of patch-changed files in the first five distinct files.
nDCG@5 uses binary file relevance. MRR uses the first relevant file anywhere in the
budgeted ranking; archived `retrieved_files` contain only the first five. A miss at
five can therefore still have nonzero MRR. Retrieval latency excludes checkout and
index construction; new runs report setup time separately.

## Dense retrieval and cache

CodeBERT is a general code encoder, not a repository-retrieval model fine-tuned by
this project. Inputs are truncated at 512 subword tokens and pooled with the attention
mask; long source chunks may lose tail content. This is a baseline design choice.

The encoder is shared across tasks. Cache keys include the immutable resolved model
revision and the ordered chunk IDs, paths, text and pooling/truncation protocol. A
changed file, changed weights or changed chunk order invalidates the index. Encoders
without a resolved revision are not cached. Queries are encoded afresh, and cached
repository embeddings do not contain gold labels. Cache writes are atomic; trusted
local cache tensors are loaded with `weights_only=True`.

Dense and hybrid methods share the index. Hybrid retrieves four times the requested
chunk pool from each method and combines ranks with constant 60, then applies the
same downstream packing. Its two previously divergent implementations were consolidated.

## Failure analysis and study limits

The audit verifies archived Recall/nDCG against the exact supplied patch labels
before reporting examples. It records missed paths rather than inventing semantic
root causes. For example, retrieving a test file but missing the implementation is
an observed localization failure; it does not establish why the ranking failed.

The slice is small and repository-imbalanced. Hash-based selection avoids handpicking
successful issues, but does not establish representativeness. Task-bootstrap intervals
ignore within-repository dependence. Further work should freeze a larger stratified
slice before tuning chunk selection or fitting a retriever.

The original dense comparison did not complete. Optimization and real-model smoke
validation improve its engineering readiness, not its measured research performance.
A checkpoint is partial evidence, not automatic resume or a completed benchmark.
