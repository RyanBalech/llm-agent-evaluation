import json

import pytest

from llm_agent_eval.explanations import explain_result, validate_explanation
from llm_agent_eval.providers import Usage

EVIDENCE = [{"id": "orders.py", "path": "orders.py", "start_line": 1, "end_line": 2,
             "text": 'order["status"] = "cancelled"\nreturn order', "source_url": None}]


def valid():
    return {"findings": [{"statement": "The excerpt changes status and returns the order.",
                          "evidence": [{"id": "orders.py", "quote": 'order["status"] = "cancelled"'}]}],
            "suggested_check": "Check whether the caller restores inventory."}


def test_citations_attach_only_server_supplied_source_metadata():
    payload = valid()
    payload["findings"][0]["evidence"][0]["source_url"] = "https://evil.test"
    payload["findings"][0]["evidence"][0]["quote"] = "invented source code"
    result = validate_explanation(json.dumps(payload), EVIDENCE)
    assert result["findings"][0]["evidence"][0]["source_url"] is None
    assert result["findings"][0]["evidence"][0]["path"] == "orders.py"
    assert result["findings"][0]["evidence"][0]["quote"] == EVIDENCE[0]["text"]


@pytest.mark.parametrize("change", ["unknown_id", "no_evidence"])
def test_rejects_unsupported_citations(change):
    payload = valid()
    if change == "unknown_id":
        payload["findings"][0]["evidence"][0]["id"] = "E99"
    else:
        payload["findings"][0]["evidence"] = []
    with pytest.raises(ValueError):
        validate_explanation(json.dumps(payload), EVIDENCE)


def test_explanation_uses_only_retrieved_evidence_and_records_provenance():
    class FakeClient:
        model_name = "test-model"
        revision = "test-revision"
        last_usage = Usage(100, 30, 130)
        def complete(self, prompt):
            assert 'order[\\"status\\"]' in prompt
            assert "gold_files" not in prompt
            return json.dumps(valid())
    result = {"files": [{"snippets": [{**EVIDENCE[0], "chunk_id": "orders.py:1-2"}]}],
              "index_id": "abc", "query": "Cancellation fails", "method": "bm25", "commit": "123"}
    explained = explain_result(result, FakeClient())
    assert explained["model_revision"] == "test-revision"
    assert explained["usage"]["total_tokens"] == 130
    assert explained["commit"] == "123"
