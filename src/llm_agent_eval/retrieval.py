from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .indexing import CodeChunk

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RankedChunk:
    chunk: CodeChunk
    score: float


class BM25Retriever:
    """Small dependency-free BM25 baseline suitable for controlled experiments."""

    def __init__(self, chunks: list[CodeChunk], *, k1: float = 1.5, b: float = 0.75,
                 path_weight: float = 0.15):
        if not chunks:
            raise ValueError("Cannot index an empty repository")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.path_weight = path_weight
        self.docs = [tokenize(f"{c.path}\n{c.text}") for c in chunks]
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        document_frequency = Counter()
        for doc in self.docs:
            document_frequency.update(set(doc))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 10,
               path_weight: float | None = None) -> list[RankedChunk]:
        if top_k <= 0:
            return []
        query_terms = tokenize(query)
        path_weight = self.path_weight if path_weight is None else path_weight
        query_set = set(query_terms)
        scored: list[RankedChunk] = []

        for chunk, doc, tf in zip(self.chunks, self.docs, self.term_freqs):
            score = 0.0
            dl = len(doc)
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denominator

            path_terms = set(tokenize(chunk.path))
            score += path_weight * len(query_set & path_terms)
            scored.append(RankedChunk(chunk, score))

        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]


def unique_file_ranking(results: list[RankedChunk]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk.path not in seen:
            seen.add(result.chunk.path)
            files.append(result.chunk.path)
    return files
