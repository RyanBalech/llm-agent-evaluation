from llm_agent_eval.swebench import (\n    from_mapping,\n    gold_files_from_patch,\n)


PATCH = """diff --git a/pkg/core.py b/pkg/core.py
index 111..222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_core.py b/tests/test_core.py
index 333..444 100644
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-old
+new
"""


def test_gold_files_are_derived_from_patch_for_scoring_only():
    assert gold_files_from_patch(PATCH) == {"pkg/core.py", "tests/test_core.py"}


def test_swebench_mapping_preserves_revision_and_problem():
    record = from_mapping({
        "instance_id": "repo__issue-1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "Something fails",
        "patch": PATCH,
    })
    assert record.base_commit == "abc123"
    assert record.gold_files == {"pkg/core.py", "tests/test_core.py"}
