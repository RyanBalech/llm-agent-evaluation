# Open questions

The current evidence and failure examples are in [RETRIEVAL_STUDY.md](RETRIEVAL_STUDY.md).

The next measured comparison is dense/hybrid versus lexical retrieval on the same
frozen task list. Complete all requested tasks before comparing scores; partial
checkpoints are for recovery and diagnosis only.

The failures motivate a file-diversity packing ablation and test/implementation
confusion analysis. Freeze a larger, repository-stratified slice before tuning those
choices. Character budgets need an explicit tokenizer before claiming LLM token costs.

LLM reranking, iterative tool use and patch validation remain future experiments.
Their interfaces do not constitute demonstrated agent performance.
