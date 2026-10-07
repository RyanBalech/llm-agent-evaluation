from llm_agent_eval.hybrid import HybridRetriever
from llm_agent_eval.indexing import CodeChunk
from llm_agent_eval.retrieval import RankedChunk


class FakeRetriever:
    def __init__(self, names):
        self.names = names

    def search(self, query, *, top_k=10):
        return [
            RankedChunk(CodeChunk(f"{name}:1-1", name, 1, 1, "x"), 1.0 / (i + 1))
            for i, name in enumerate(self.names[:top_k])
        ]


def test_hybrid_promotes_shared_candidate():
    hybrid = HybridRetriever(
        FakeRetriever(["a.py", "b.py"]),
        FakeRetriever(["b.py", "c.py"]),
        candidate_multiplier=1,
    )
    assert hybrid.search("bug", top_k=2)[0].chunk.path == "b.py"
