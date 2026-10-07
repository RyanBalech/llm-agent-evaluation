# Experiment protocol

## Leakage boundary

Reference patches and gold files are evaluation labels. They must never appear in retrieval
queries, prompts, indexes, or model-visible metadata.

## Fair comparisons

For a comparison to be labelled budget-matched:

1. use the same benchmark task subset;
2. use the same repository revision;
3. enforce the same maximum downstream context budget;
4. keep the generation model and decoding settings fixed when measuring retrieval effects;
5. log failures instead of silently dropping them.

## Reproducibility

Every reported run should record:

- git commit SHA;
- dataset/benchmark version;
- task IDs;
- model identifier;
- retrieval configuration;
- random seed where relevant;
- token/context limits;
- wall-clock latency;
- package/runtime versions.

## Statistics

Report task-level metrics and aggregate means. For model comparisons, compute paired differences
and bootstrap confidence intervals over tasks. Predefine the primary metric before interpreting
results.

## Benchmark progression

Start with deterministic local tests, then a small development slice, then freeze configuration
before evaluating the held-out benchmark slice. SWE-bench Verified is the intended external
benchmark once the adapter is implemented.

## Reporting rule

A README table may contain benchmark numbers only if the command/configuration required to
reproduce them is committed and the corresponding result artifact is stored or linked.
