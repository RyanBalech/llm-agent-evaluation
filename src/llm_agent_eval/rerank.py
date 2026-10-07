from __future__ import annotations

import json
from typing import Protocol

from .retrieval import RankedChunk


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def build_rerank_prompt(issue: str, candidates: list[RankedChunk]) -> str:
    blocks = []
    for item in candidates:
        c = item.chunk
        blocks.append(
            f"ID: {c.chunk_id}\nPATH: {c.path}\nLINES: {c.start_line}-{c.end_line}\n{c.text}"
        )
    joined = "\n\n---\n\n".join(blocks)
    return (
        "You are localizing a software issue. Rank only the candidate chunk IDs by how likely "
        "they are to contain code that must change. Return strict JSON with one key, "
        '\"ranked_chunk_ids\", whose value is an array of IDs. Do not propose a patch.\n\n'
        f"ISSUE:\n{issue}\n\nCANDIDATES:\n{joined}"
    )


def rerank(issue: str, candidates: list[RankedChunk], client: LLMClient) -> list[RankedChunk]:
    by_id = {item.chunk.chunk_id: item for item in candidates}
    response = json.loads(client.complete(build_rerank_prompt(issue, candidates)))
    ids = response.get("ranked_chunk_ids")
    if not isinstance(ids, list):
        raise ValueError("LLM response must contain ranked_chunk_ids")

    ordered: list[RankedChunk] = []
    seen: set[str] = set()
    for chunk_id in ids:
        if chunk_id in by_id and chunk_id not in seen:
            ordered.append(by_id[chunk_id])
            seen.add(chunk_id)

    ordered.extend(item for item in candidates if item.chunk.chunk_id not in seen)
    return ordered
