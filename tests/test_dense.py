from llm_agent_eval.dense import reciprocal_rank_fusion
from llm_agent_eval.indexing import CodeChunk
from llm_agent_eval.retrieval import RankedChunk


def item(name: str, score: float) -> RankedChunk:
    return RankedChunk(CodeChunk(f"{name}:1-1", name, 1, 1, "x"), score)


def test_rrf_rewards_chunks_supported_by_multiple_retrievers():
    sparse = [item("a.py", 2), item("b.py", 1)]
    dense = [item("b.py", 0.9), item("c.py", 0.8)]
    fused = reciprocal_rank_fusion([sparse, dense], top_k=3)
    assert fused[0].chunk.path == "b.py"
