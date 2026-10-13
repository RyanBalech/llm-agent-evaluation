"""Browser adapter for the same Python lexical retriever used by the server."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from time import perf_counter
from urllib.parse import quote

from .evaluation import select_under_character_budget
from .indexing import CodeChunk
from .retrieval import BM25Retriever, tokenize


class BrowserIndex:
    def __init__(self, snapshot: dict):
        self.row = snapshot["index"]
        self.retriever = BM25Retriever([CodeChunk(**c) for c in snapshot["chunks"]])

    def search(self, payload: dict) -> dict:
        if payload.get("method", "bm25") != "bm25":
            raise ValueError("The browser demo supports BM25; run the server for dense retrieval")
        query = payload["query"].strip()
        budget, top_k = payload.get("context_budget_chars", 40000), payload.get("top_k", 5)
        if not 3 <= len(query) <= 8000 or not 1000 <= budget <= 80000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid search query or limits")
        started = perf_counter()
        candidates = [c for c in self.retriever.search(query, top_k=50) if c.score > 0]
        retrieval_ms = (perf_counter() - started) * 1000
        selected = select_under_character_budget(candidates, budget)
        files = OrderedDict()
        terms = set(tokenize(query))
        row = self.row
        for item in selected:
            chunk = item.chunk
            if chunk.path not in files:
                if len(files) >= top_k:
                    continue
                files[chunk.path] = {"path": chunk.path, "score": round(item.score, 6), "snippets": []}
            url = None if row["repository"] == "demo" else (
                f'https://github.com/{row["repository"]}/blob/{row["commit"]}/'
                f'{quote(chunk.path, safe="/")}#L{chunk.start_line}-L{chunk.end_line}')
            files[chunk.path]["snippets"].append({**asdict(chunk), "source_url": url,
                "matched_terms": sorted(terms & set(tokenize(chunk.path + "\n" + chunk.text)))})
        return {"index_id": row["id"], "repository": row["repository"], "commit": row["commit"],
                "query": query, "method": "bm25", "context_budget_chars": budget,
                "selected_characters": sum(len(c.chunk.text) for c in selected),
                "candidate_chunks": len(candidates), "selected_chunks": len(selected),
                "retrieval_ms": round(retrieval_ms, 2),
                "total_ms": round((perf_counter() - started) * 1000, 2),
                "files": list(files.values()), "index_sha256": row["index_sha256"],
                "configuration": {"lines_per_chunk": 80, "overlap": 20, "candidate_limit": 50,
                                  "path_prior": 0.15, "top_k": top_k},
                "execution": "Python BM25 in a browser WebAssembly worker",
                "explanation": "Matched terms are lexical evidence, not proof of a bug. "
                               "Scores are ranking signals, not probabilities. Inspect the cited code."}
