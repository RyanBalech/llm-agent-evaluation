# Repository-Level LLM Agent Evaluation

[![CI](https://github.com/RyanBalech/llm-agent-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/RyanBalech/llm-agent-evaluation/actions/workflows/ci.yml)

A research-oriented framework for studying **how repository retrieval and structured tool use affect software-engineering agents under fixed context and cost budgets**.

The project starts from a deliberately narrow question:

> **How much does repository-level retrieval improve fault localization before an LLM ever attempts a patch?**

That separation matters. End-to-end coding-agent scores mix together retrieval, reasoning, patch generation, tool use, and test feedback. This repository makes those components measurable independently before composing them into a full agent.

## Research status

| Component | Status |
|---|---|
| Repository chunking and indexing | Implemented |
| BM25 lexical retrieval baseline | Implemented |
| Localization metrics (Recall@k, MRR, nDCG) | Implemented |
| Context-budget accounting | Implemented |
| Structured LLM reranking interface | Implemented |
| Deterministic tests + CI | Implemented |
| PyTorch Transformer dense retrieval | Implemented |
| SWE-bench patch-label adapter | Implemented |
| Patch generation + validation | Planned |
| Agent/tool-use ablations | Planned |
| 20-instance SWE-bench Verified BM25 study | Completed |
| Dense + hybrid matched comparison | Implemented; benchmark run pending |

**Current measured baseline:** on the frozen 20-instance SWE-bench Verified study slice (excluding the three engineering smoke tasks), BM25 achieved Recall@5 **0.425**, MRR **0.279**, and nDCG@5 **0.297**, with 20/20 tasks completed. This is a controlled localization study, not a full SWE-bench patch-resolution score. See `docs/RETRIEVAL_STUDY.md` for the protocol and limitations.

## Why localization first?

Recent software-engineering agent work shows that strong results do not necessarily require a large autonomous control loop. A simpler pipeline can first localize the relevant repository context, then reason over a much smaller candidate set. This project treats localization as a first-class research problem rather than an invisible prompt-preparation step.

The initial experiments compare:

1. **No retrieval** — repository context ordered without issue-aware ranking.
2. **BM25** — sparse lexical retrieval over code chunks.
3. **BM25 + path prior** — lexical retrieval with a small filename/path relevance prior.
4. **BM25 + LLM reranking** — retrieve a larger candidate set, then ask an LLM to return a structured ranking.
5. **Dense / hybrid retrieval** — Transformer embeddings in PyTorch plus Reciprocal Rank Fusion.
6. **Tool-use agent** — planned extension where the model can search/read files iteratively.

All methods are evaluated under the same context budget.

## Core metrics

For localization we track:

- **Recall@k** — whether at least one gold file appears in the top-k retrieved files.
- **MRR** — reciprocal rank of the first relevant file.
- **nDCG@k** — rewards ranking multiple relevant files near the top.
- **Context size** — characters / estimated tokens sent downstream.
- **Latency** — retrieval and reranking time.
- **LLM usage** — request count and provider-reported token usage when available.

The goal is not just to maximize accuracy. The interesting question is the **quality–cost frontier**.

## Repository layout

```text
.
├── src/llm_agent_eval/
│   ├── indexing.py       # repository scanning and line-based code chunks
│   ├── retrieval.py      # BM25 + path-aware ranking
│   ├── rerank.py         # provider-agnostic structured LLM reranking
│   ├── metrics.py        # localization metrics
│   ├── evaluation.py     # experiment runner
│   ├── datasets.py       # JSONL task loading
│   └── cli.py            # command-line entry point
├── tests/                # deterministic unit/integration tests
├── examples/             # tiny local benchmark for smoke tests
├── configs/              # experiment configurations
└── docs/
    ├── RESEARCH_PLAN.md
    └── EXPERIMENT_PROTOCOL.md
```

## Quick start

```bash
git clone https://github.com/RyanBalech/llm-agent-evaluation.git
cd llm-agent-evaluation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q
```

Optional deep-retrieval dependencies:

```bash
pip install -e ".[ml]"
```

Run the deterministic toy localization benchmark:

```bash
llm-agent-eval evaluate \
  --repo examples/toy_repo \
  --tasks examples/toy_tasks.jsonl \
  --top-k 3
```

The output is JSON so runs can be logged or compared automatically.

## Task format

Each benchmark task is a JSON object:

```json
{
  "task_id": "toy-divide-zero",
  "issue": "Division by zero should return a clear validation error instead of crashing.",
  "gold_files": ["calculator.py"]
}
```

For SWE-bench-style experiments, `gold_files` can be derived from the reference patch **only for evaluation**. The retriever never receives it.

## LLM reranking

The reranking layer is deliberately provider-agnostic. Any client implementing:

```python
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...
```

can be used. The model is asked to return strict JSON containing chunk IDs, making the output auditable and easy to score.

A Mistral SDK adapter is implemented while the research interface remains provider-agnostic; credentials are supplied only through environment variables.

## Experimental principles

This project follows five rules:

1. **Time/order-aware evaluation where applicable.**
2. **No gold-patch leakage into retrieval or prompts.**
3. **Equal context budgets across methods.**
4. **Paired comparisons on the same benchmark instances.**
5. **Report uncertainty and failure cases, not only mean scores.**

See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Planned research questions

- Does an LLM reranker materially improve file localization over BM25 at the same context budget?
- When does path-aware lexical retrieval beat dense retrieval for repository search?
- How quickly do localization gains saturate with larger candidate sets?
- Does iterative tool use help beyond one-shot retrieval after controlling for tokens and latency?
- Are gains stable across repository size, issue length, and bug type?
- How much end-to-end patch success can be explained by localization quality alone?

## Positioning

The project is intended as a reproducible research artifact, not a production coding agent. The final target is a benchmark-backed report with:

- strong classical baselines,
- modern LLM/agent variants,
- controlled ablations,
- statistical uncertainty,
- cost/latency analysis,
- failure taxonomy,
- and fully reproducible code.



## Comparative retrieval experiments

The repository now includes a common evaluation path for BM25, Transformer dense retrieval, and sparse-dense reciprocal-rank fusion. The matched comparison fixes task set, candidate count, top-k, and downstream character budget, then reports paired bootstrap confidence intervals over task-level Recall@k, MRR, and nDCG.

A separate context-budget ablation evaluates the three retrievers at 10k, 20k, 40k, and 80k characters to measure the quality-context frontier rather than relying on a single arbitrary prompt budget.
