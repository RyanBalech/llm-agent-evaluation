from __future__ import annotations

from .dense import reciprocal_rank_fusion
from .retrieval import RankedChunk


class HybridRetriever:
    """Reciprocal-rank fusion of sparse and dense candidate rankings."""

    def __init__(self, sparse, dense, *, candidate_multiplier: int = 4, rrf_constant: int = 60):
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self.sparse = sparse
        self.dense = dense
        self.candidate_multiplier = candidate_multiplier
        self.rrf_constant = rrf_constant

    def search(self, query: str, *, top_k: int = 10) -> list[RankedChunk]:
        if top_k <= 0:
            return []
        n = top_k * self.candidate_multiplier
        sparse = self.sparse.search(query, top_k=n)
        dense = self.dense.search(query, top_k=n)
        return reciprocal_rank_fusion(
            [sparse, dense], top_k=top_k, constant=self.rrf_constant
        )
