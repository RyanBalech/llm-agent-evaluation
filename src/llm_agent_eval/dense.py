from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .indexing import CodeChunk
from .retrieval import RankedChunk


def embedding_cache_key(chunks: list[CodeChunk], *, model_name: str, revision: str) -> str:
    """Identify ordered model inputs and encoder weights, not a checkout directory."""
    payload = {"encoder": model_name, "revision": revision,
               "pooling": "attention-masked-mean+l2", "max_length": 512,
               "documents": [(chunk.chunk_id, chunk.path, chunk.text) for chunk in chunks]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class TransformerEncoder:
    """Load encoder weights once per benchmark; each repository gets its own index."""
    def __init__(self, model_name: str, *, revision: str | None = None,
                 batch_size: int = 16, device: str | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.batch_size = model_name, batch_size
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModel.from_pretrained(model_name, revision=revision)
        self.revision = getattr(self._model.config, "_commit_hash", None)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def encode(self, texts: list[str]):
        torch = self._torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    texts[start : start + self.batch_size], padding=True,
                    truncation=True, max_length=512, return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(batches, dim=0)


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
    revision: str | None = None
    cache_dir: str | Path | None = None
    encoder: TransformerEncoder | None = None

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("Cannot index an empty repository")

        import torch
        self._torch = torch
        self.encoder = self.encoder or TransformerEncoder(
            self.model_name, revision=self.revision, batch_size=self.batch_size, device=self.device)
        if self.encoder.model_name != self.model_name:
            raise ValueError("shared encoder must match model_name")
        self.resolved_revision = self.encoder.revision
        self.cache_hit = False
        cache_path = None
        # Mutable branches/local encoders without a resolved commit are never cached.
        if self.cache_dir is not None and self.resolved_revision:
            key = embedding_cache_key(self.chunks, model_name=self.model_name,
                                      revision=self.resolved_revision)
            cache_path = Path(self.cache_dir) / f"{key}.pt"
            if cache_path.exists():
                embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
                if embeddings.ndim != 2 or len(embeddings) != len(self.chunks):
                    raise ValueError("cached embeddings do not match repository chunks")
                self._embeddings = embeddings
                self.cache_hit = True
                return
        documents = [f"{chunk.path}\n{chunk.text}" for chunk in self.chunks]
        self._embeddings = self.encoder.encode(documents)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            torch.save(self._embeddings, temporary)
            temporary.replace(cache_path)

    def search(self, query: str, *, top_k: int = 10) -> list[RankedChunk]:
        if top_k <= 0:
            return []
        query_embedding = self.encoder.encode([query])[0]
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
