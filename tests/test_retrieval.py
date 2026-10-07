from pathlib import Path

from llm_agent_eval.indexing import chunk_repository
from llm_agent_eval.retrieval import BM25Retriever, unique_file_ranking


def test_bm25_localizes_relevant_file(tmp_path: Path):
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n    return token is not None\n", encoding="utf-8"
    )
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    chunks = chunk_repository(tmp_path, lines_per_chunk=10, overlap=0)
    results = BM25Retriever(chunks).search("token validation fails in authentication", top_k=2)
    assert unique_file_ranking(results)[0] == "auth.py"


def test_invalid_chunk_configuration_is_rejected(tmp_path: Path):
    try:
        chunk_repository(tmp_path, lines_per_chunk=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
