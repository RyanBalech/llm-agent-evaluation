from llm_agent_eval.benchmark import load_swebench_jsonl
from llm_agent_eval.swebench import SWEBenchRecord


def test_jsonl_loader_preserves_gold_labels(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"instance_id":"x-1","repo":"o/r","base_commit":"abc",'
        '"problem_statement":"bug","patch":"diff --git a/a.py b/a.py\\n"}\\n',
        encoding="utf-8",
    )
    records = load_swebench_jsonl(path)
    assert records[0].instance_id == "x-1"
    assert records[0].gold_files == {"a.py"}


def test_record_type_keeps_reference_patch_separate():
    record = SWEBenchRecord("x", "o/r", "abc", "visible issue", "secret patch")
    assert record.problem_statement == "visible issue"
    assert record.patch == "secret patch"
