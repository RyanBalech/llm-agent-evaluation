import json

from llm_agent_eval.indexing import CodeChunk
from llm_agent_eval.rerank import rerank
from llm_agent_eval.retrieval import RankedChunk


class FakeClient:
    def complete(self, prompt: str) -> str:
        assert "Do not propose a patch" in prompt
        return json.dumps({"ranked_chunk_ids": ["b.py:1-1"]})


def test_reranker_respects_structured_ranking_and_preserves_omitted_candidates():
    candidates = [
        RankedChunk(CodeChunk("a.py:1-1", "a.py", 1, 1, "pass"), 2.0),
        RankedChunk(CodeChunk("b.py:1-1", "b.py", 1, 1, "fix = True"), 1.0),
    ]
    ranked = rerank("fix bug", candidates, FakeClient())
    assert [x.chunk.path for x in ranked] == ["b.py", "a.py"]
