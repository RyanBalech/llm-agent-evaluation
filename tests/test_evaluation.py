from llm_agent_eval.evaluation import select_under_character_budget
from llm_agent_eval.indexing import CodeChunk
from llm_agent_eval.retrieval import RankedChunk


def ranked(name: str, text: str) -> RankedChunk:
    return RankedChunk(CodeChunk(f"{name}:1-1", name, 1, 1, text), 1.0)


def test_character_budget_never_exceeds_limit():
    candidates = [ranked("a.py", "12345"), ranked("b.py", "1234"), ranked("c.py", "123")]
    selected = select_under_character_budget(candidates, 8)
    assert sum(len(x.chunk.text) for x in selected) <= 8
    assert [x.chunk.path for x in selected] == ["a.py", "c.py"]
