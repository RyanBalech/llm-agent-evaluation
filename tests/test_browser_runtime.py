import time
from dataclasses import asdict
from pathlib import Path

from llm_agent_eval.browser_runtime import BrowserIndex
from llm_agent_eval.indexing import chunk_repository
from llm_agent_eval.service import IndexService


def test_browser_adapter_matches_server_ranking_budget_and_evidence(tmp_path):
    service = IndexService(tmp_path)
    try:
        row = service.submit("demo")
        for _ in range(100):
            row = service.get(row["id"])
            if row["status"] == "ready":
                break
            time.sleep(.02)
        assert row["status"] == "ready"
        chunks = chunk_repository(Path(__file__).parents[1] / "src/llm_agent_eval/demo")
        browser = BrowserIndex({"index": row, "chunks": [asdict(c) for c in chunks]})
        query = "Cancelling an order does not restore inventory."
        for budget in (1000, 4000):
            expected = service.search(row["id"], query, "bm25", budget, 3)
            actual = browser.search({"query": query, "context_budget_chars": budget, "top_k": 3})
            for key in ("files", "selected_characters", "candidate_chunks", "selected_chunks", "configuration"):
                assert actual[key] == expected[key]
    finally:
        service.close()
