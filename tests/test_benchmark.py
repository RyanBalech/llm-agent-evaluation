import json

from llm_agent_eval.benchmark import load_swebench_jsonl
from llm_agent_eval.swebench import SWEBenchRecord


def test_jsonl_loader_preserves_gold_labels(tmp_path):
    path = tmp_path / "tasks.jsonl"
    record = {
        "instance_id": "x-1",
        "repo": "o/r",
        "base_commit": "abc",
        "problem_statement": "bug",
        "patch": "diff --git a/a.py b/a.py\n",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    records = load_swebench_jsonl(path)
    assert records[0].instance_id == "x-1"
    assert records[0].gold_files == {"a.py"}


def test_record_type_keeps_reference_patch_separate():
    record = SWEBenchRecord("x", "o/r", "abc", "visible issue", "secret patch")
    assert record.problem_statement == "visible issue"
    assert record.patch == "secret patch"


def test_checkpoint_preserves_completed_tasks_and_failures(tmp_path, monkeypatch):
    from llm_agent_eval import benchmark

    checkpoint = tmp_path / "result.json"
    records = [SWEBenchRecord(str(i), "o/r", "abc", "calculator bug",
                             "diff --git a/calculator.py b/calculator.py\n") for i in range(2)]

    def checkout(record, destination):
        if record.instance_id == "1":
            partial = json.loads(checkpoint.read_text())
            assert partial["n_completed"] == 1
            assert partial["n_requested"] == 2
            raise RuntimeError("checkout failed")
        repo = destination / record.instance_id
        repo.mkdir()
        (repo / "calculator.py").write_text("def calculator(): return 1\n")
        return repo

    monkeypatch.setattr(benchmark, "checkout_revision", checkout)
    result = benchmark.evaluate_records(records, checkpoint_path=checkpoint)
    assert result["n_successful"] == 1
    assert result["n_failed"] == 1
    assert json.loads(checkpoint.read_text()) == result
