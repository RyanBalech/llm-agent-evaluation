from __future__ import annotations

from dataclasses import dataclass

from .indexing import CodeChunk
from .retrieval import RankedChunk


@dataclass
class TransformerDenseRetriever:
    """Dense code retriever using a Hugging Face encoder and PyTorch cosine similarity.

    The dependency is optional: install with `pip install -e ".[ml]"`.
    A model identifier is always explicit so benchmark runs can record the exact encoder.
    """

    chunks: list[CodeChunk]
    model_name: str
    batch_size: int = 16
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("Cannot index an empty repository")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        documents = [f"{chunk.path}\n{chunk.text}" for chunk in self.chunks]
        self._embeddings = self._encode(documents)

    def _encode(self, texts: list[str]):
        torch = self._torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu())
        return torch.cat(batches, dim=0)

    def search(self, query: str, *, top_k: int = 10) -> list[RankedChunk]:
        if top_k <= 0:
            return []
        query_embedding = self._encode([query])[0]
        scores = self._embeddings @ query_embedding
        indices = self._torch.argsort(scores, descending=True)[:top_k].tolist()
        return [RankedChunk(self.chunks[i], float(scores[i])) for i in indices]


def reciprocal_rank_fusion(
    rankings: list[list[RankedChunk]], *, top_k: int = 10, constant: int = 60
) -> list[RankedChunk]:
    """Fuse heterogeneous rankings without requiring calibrated score scales."""
    if constant <= 0:
        raise ValueError("constant must be positive")

    by_id: dict[str, RankedChunk] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            by_id.setdefault(chunk_id, item)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [RankedChunk(by_id[chunk_id].chunk, scores[chunk_id]) for chunk_id in ordered]
