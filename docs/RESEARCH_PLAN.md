# Research plan

## Central hypothesis

Repository-aware retrieval can improve software-issue localization while reducing the amount of
code an LLM must inspect. The key test is whether those gains remain after controlling for the
context budget, model, and benchmark instances.

## Study 1 — Retrieval baselines

Compare no issue-aware ranking, BM25, path-aware BM25, dense retrieval, and hybrid retrieval.
Primary outcomes: Recall@k, MRR, nDCG@k, context size, and latency.

## Study 2 — LLM reranking

Retrieve a fixed candidate pool, rerank it with an instruction-tuned model, and compare against
the original ranking. Hold candidate pool and downstream context budget fixed. Report paired
per-task differences and bootstrap confidence intervals.

## Study 3 — Iterative tool use

Give the model search/read tools and the same total token budget. Compare one-shot retrieval
against iterative search. Track tool calls, tokens, wall time, localization quality, and failure
modes.

## Study 4 — Patch generation

Only after localization experiments are stable, add patch generation and test execution.
Measure resolved rate alongside localization quality to study how strongly localization predicts
end-to-end success.

## Ablations

- chunk size and overlap
- candidate pool size
- path prior on/off
- query formed from title vs full issue
- context budget
- reranker model
- one-shot vs iterative retrieval

## Analysis

Use paired comparisons because every method is evaluated on the same tasks. Report bootstrap
confidence intervals and per-repository breakdowns. Do not treat overlapping confidence intervals
as a hypothesis test. Include a failure taxonomy with concrete examples.
