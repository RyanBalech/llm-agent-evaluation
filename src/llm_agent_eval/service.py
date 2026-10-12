"""Single-process application service: durable job metadata and immutable source indexes."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.parse import quote

from .evaluation import select_under_character_budget
from .indexing import CodeChunk, chunk_repository
from .repository import fetch_sources
from .retrieval import BM25Retriever, tokenize

LOG = logging.getLogger("repolens")
MODEL = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


class CapacityError(ValueError):
    pass


class IndexService:
    def __init__(self, root: Path, *, enable_dense: bool = False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.enable_dense = enable_dense
        self.db = self.root / "repolens.sqlite3"
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.cache: OrderedDict = OrderedDict()
        self.encoder = None
        with self.connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS indexes (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        for row in self.list():
            if row["status"] in {"queued", "indexing"}:
                self.save({**row, "status": "failed", "error": "Server restarted; submit again"})
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="repolens-index")

    def connect(self):
        return sqlite3.connect(self.db, timeout=10)

    def save(self, row: dict):
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO indexes VALUES (?, ?)",
                        (row["id"], json.dumps(row)))

    def list(self) -> list[dict]:
        with self.connect() as con:
            rows = [json.loads(row[0]) for row in con.execute("SELECT data FROM indexes")]
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def get(self, index_id: str) -> dict:
        with self.connect() as con:
            result = con.execute("SELECT data FROM indexes WHERE id = ?", (index_id,)).fetchone()
        if not result:
            raise KeyError(index_id)
        return json.loads(result[0])

    def submit(self, repository: str, ref: str = "", *, dense: bool = False) -> dict:
        if dense and not self.enable_dense:
            raise ValueError("Dense indexing is disabled on this server")
        with self.lock:
            rows = self.list()
            if repository == "demo":
                for row in rows:
                    if row["repository"] == "demo" and row["status"] != "failed":
                        return row
            if sum(r["status"] in {"queued", "indexing"} for r in rows) >= 3:
                raise CapacityError("Index queue is full; wait for a job to finish")
            if len(rows) >= 20:
                raise CapacityError("Storage limit reached; delete an old index first")
            row = {"id": uuid.uuid4().hex, "repository": repository, "requested_ref": ref,
                   "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
                   "methods": ["bm25", "dense", "hybrid"] if dense else ["bm25"],
                   "error": None}
            self.save(row)
            self.worker.submit(self.build, row)
            return row

    def build(self, row: dict):
        started = perf_counter()
        row = {**row, "status": "indexing"}
        self.save(row)
        try:
            if row["repository"] == "demo":
                chunks = chunk_repository(Path(__file__).parent / "demo")
                commit = "demo-" + hashlib.sha256(
                    "".join(c.text for c in chunks).encode()).hexdigest()[:12]
                stats = {"source_bytes": sum(len(c.text.encode()) for c in chunks),
                         "skipped_files": 0}
            else:
                with TemporaryDirectory(prefix="source-", dir=self.root) as tmp:
                    chunks, commit, stats = fetch_sources(
                        row["repository"], row["requested_ref"], Path(tmp) / "repo")
            retrievers = self.make_retrievers(chunks, "dense" in row["methods"])
            payload = {"schema_version": 1, "commit": commit,
                       "chunks": [asdict(c) for c in chunks]}
            path = self.root / f'{row["id"]}.json'
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
            row.update(status="ready", commit=commit, files=len({c.path for c in chunks}),
                       chunks=len(chunks), indexing_seconds=round(perf_counter() - started, 3),
                       index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **stats)
            if "dense" in row["methods"]:
                row.update(model=MODEL, model_revision=MODEL_REVISION)
            with self.lock:
                self.cache_put(row["id"], retrievers)
                self.save(row)
            LOG.info(json.dumps({"event": "index_ready", "index_id": row["id"],
                                 "files": row["files"], "seconds": row["indexing_seconds"]}))
        except Exception as exc:
            LOG.exception("Index failed: %s", row["id"])
            message = str(exc) if isinstance(exc, ValueError) else "Indexing failed; check server logs"
            self.save({**row, "status": "failed", "error": message})

    def make_retrievers(self, chunks: list[CodeChunk], dense: bool) -> dict:
        result = {"bm25": BM25Retriever(chunks)}
        if dense:
            from .dense import TransformerDenseRetriever, TransformerEncoder
            from .hybrid import HybridRetriever
            with self.inference_lock:
                self.encoder = self.encoder or TransformerEncoder(MODEL, revision=MODEL_REVISION)
                result["dense"] = TransformerDenseRetriever(
                    chunks, model_name=MODEL, encoder=self.encoder, cache_dir=self.root / "embeddings")
                result["hybrid"] = HybridRetriever(result["bm25"], result["dense"])
        return result

    def cache_put(self, index_id: str, retrievers: dict):
        self.cache[index_id] = retrievers
        self.cache.move_to_end(index_id)
        while len(self.cache) > 3:
            self.cache.popitem(last=False)

    def delete(self, index_id: str):
        with self.lock:
            row = self.get(index_id)
            if row["status"] in {"queued", "indexing"}:
                raise ValueError("Wait for indexing to finish before deleting")
            (self.root / f"{index_id}.json").unlink(missing_ok=True)
            self.cache.pop(index_id, None)
            with self.connect() as con:
                con.execute("DELETE FROM indexes WHERE id = ?", (index_id,))

    def search(self, index_id: str, query: str, method: str, budget: int, top_k: int) -> dict:
        started = perf_counter()
        with self.lock:
            row = self.get(index_id)
            if row["status"] != "ready":
                raise ValueError("This index is not ready")
            if method not in row["methods"]:
                raise ValueError("This method is not available for this index")
            if method != "bm25" and not self.enable_dense:
                raise ValueError("Dense retrieval is disabled on this server")
            if index_id not in self.cache:
                raw = (self.root / f"{index_id}.json").read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["index_sha256"]:
                    raise ValueError("Index snapshot failed integrity verification; reindex the repository")
                payload = json.loads(raw)
                self.cache_put(index_id, self.make_retrievers(
                    [CodeChunk(**c) for c in payload["chunks"]],
                    self.enable_dense and "dense" in row["methods"]))
            retriever = self.cache[index_id][method]
            self.cache.move_to_end(index_id)
        retrieval_start = perf_counter()
        if method == "bm25":
            candidates = retriever.search(query, top_k=50)
            # Zero lexical evidence must not be presented as a relevant match.
            candidates = [c for c in candidates if c.score > 0]
        else:
            with self.inference_lock:
                candidates = retriever.search(query, top_k=50)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        selected = select_under_character_budget(candidates, budget)
        files = OrderedDict()
        query_terms = set(tokenize(query))
        for item in selected:
            chunk = item.chunk
            if chunk.path not in files:
                if len(files) >= top_k:
                    continue
                files[chunk.path] = {"path": chunk.path, "score": round(item.score, 6),
                                     "snippets": []}
            terms = sorted(query_terms & set(tokenize(chunk.path + "\n" + chunk.text)))
            url = None if row["repository"] == "demo" else (
                f'https://github.com/{row["repository"]}/blob/{row["commit"]}/'
                f'{quote(chunk.path, safe="/")}#L{chunk.start_line}-L{chunk.end_line}')
            files[chunk.path]["snippets"].append({**asdict(chunk), "matched_terms": terms,
                                                 "source_url": url})
        result = {"index_id": index_id, "repository": row["repository"], "commit": row["commit"],
                  "query": query, "method": method, "context_budget_chars": budget,
                  "selected_characters": sum(len(c.chunk.text) for c in selected),
                  "candidate_chunks": len(candidates), "selected_chunks": len(selected),
                  "retrieval_ms": round(retrieval_ms, 2),
                  "total_ms": round((perf_counter() - started) * 1000, 2),
                  "files": list(files.values()), "index_sha256": row["index_sha256"],
                  "configuration": {"lines_per_chunk": 80, "overlap": 20,
                                    "candidate_limit": 50, "path_prior": 0.15, "top_k": top_k},
                  "explanation": "Matched terms are lexical evidence, not proof of a bug. "
                  "Scores are ranking signals, not probabilities. Inspect the cited code."}
        LOG.info(json.dumps({"event": "search", "index_id": index_id, "method": method,
                             "retrieval_ms": result["retrieval_ms"], "files": len(files)}))
        return result

    def close(self):
        self.worker.shutdown(wait=True)
