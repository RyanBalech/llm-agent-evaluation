from llm_agent_eval.dense import embedding_cache_key, reciprocal_rank_fusion
from llm_agent_eval.indexing import CodeChunk
from llm_agent_eval.retrieval import RankedChunk


def item(name: str, score: float) -> RankedChunk:
    return RankedChunk(CodeChunk(f"{name}:1-1", name, 1, 1, "x"), score)


def test_rrf_rewards_chunks_supported_by_multiple_retrievers():
    sparse = [item("a.py", 2), item("b.py", 1)]
    dense = [item("b.py", 0.9), item("c.py", 0.8)]
    fused = reciprocal_rank_fusion([sparse, dense], top_k=3)
    assert fused[0].chunk.path == "b.py"


def test_embedding_cache_invalidates_content_order_and_weight_revision():
    chunks = [item("a.py", 0).chunk, item("b.py", 0).chunk]
    key = embedding_cache_key(chunks, model_name="encoder", revision="abc")
    assert key != embedding_cache_key(chunks[::-1], model_name="encoder", revision="abc")
    assert key != embedding_cache_key(chunks, model_name="encoder", revision="def")
    edited = [CodeChunk("a.py:1-1", "a.py", 1, 1, "changed"), chunks[1]]
    assert key != embedding_cache_key(edited, model_name="encoder", revision="abc")


def test_dense_cache_preserves_ranking_and_does_not_encode_documents_twice(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    from llm_agent_eval.dense import TransformerDenseRetriever

    class Encoder:
        model_name = "test-encoder"
        revision = "immutable-test-revision"

        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(texts)
            return torch.tensor([[1., 0.] if "a.py" in text or text == "query" else [0., 1.]
                                 for text in texts])

    chunks = [item("a.py", 0).chunk, item("b.py", 0).chunk]
    encoder = Encoder()
    first = TransformerDenseRetriever(chunks, "test-encoder", encoder=encoder, cache_dir=tmp_path)
    second = TransformerDenseRetriever(chunks, "test-encoder", encoder=encoder, cache_dir=tmp_path)
    assert not first.cache_hit and second.cache_hit
    assert len(encoder.calls) == 1
    assert [r.chunk.path for r in first.search("query")] == [r.chunk.path for r in second.search("query")]
