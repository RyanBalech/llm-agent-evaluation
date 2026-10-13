import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from llm_agent_eval.app import create_app
from llm_agent_eval.repository import normalize_repository, read_git_sources, validate_ref


def ready(client, index_id):
    for _ in range(200):
        row = client.get(f"/api/indexes/{index_id}").json()
        if row["status"] in {"ready", "failed"}:
            assert row["status"] == "ready", row
            return row
        time.sleep(.01)
    pytest.fail("Index job did not finish")


def test_demo_search_survives_restart_and_deletion(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert "default-src 'self'" in client.get("/").headers["content-security-policy"]
        row = client.post("/api/demo").json()
        ready(client, row["id"])
        query = {"index_id": row["id"], "query": "cancel order restore inventory",
                 "context_budget_chars": 1000}
        result = client.post("/api/search", json=query)
        assert result.status_code == 200, result.text
        result = result.json()
        assert result["selected_characters"] <= 1000
        assert result["files"][0]["path"] in {"orders.py", "inventory.py"}
        assert result["files"][0]["snippets"][0]["matched_terms"]
        assert client.post("/api/demo").json()["id"] == row["id"]
    with TestClient(create_app(tmp_path)) as client:
        restored = client.post("/api/search", json=query).json()
        assert restored["files"] == result["files"]
        assert restored["commit"] == result["commit"]
        assert client.delete(f'/api/indexes/{row["id"]}').status_code == 204
        assert client.post("/api/search", json=query).status_code == 404
        assert not (tmp_path / f'{row["id"]}.json').exists()


@pytest.mark.parametrize("repository", ["http://github.com/a/b", "https://evil.com/a/b",
                                        "file:///etc/passwd", "a/../b", "--help", "a/b?x=1",
                                        "https://user:password@github.com/a/b"])
def test_reject_non_github_inputs(repository):
    with pytest.raises(ValueError):
        normalize_repository(repository)


@pytest.mark.parametrize("ref", ["--upload-pack=evil", "../secret", "branch:other", "a\nb"])
def test_reject_unsafe_ref(ref):
    with pytest.raises(ValueError):
        validate_ref(ref)


def test_auth_origin_and_validation(tmp_path):
    with TestClient(create_app(tmp_path, token="test-secret")) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/indexes").status_code == 401
        client.headers["Authorization"] = "Bearer test-secret"
        assert client.get("/api/indexes").status_code == 200
        assert client.post("/api/demo", headers={"Origin": "https://evil.com"}).status_code == 403
        assert client.get("/", headers={"Host": "evil.com"}).status_code == 400
        assert client.post("/api/indexes", json={"repository": "file:///etc/passwd"}).status_code == 422
        assert client.post("/api/search", json={"index_id": "../secret", "query": "abc"}).status_code == 422
        assert client.post("/api/indexes", json={"repository": "a/b", "dense": True}).status_code == 409


def test_zero_evidence_does_not_produce_arbitrary_files(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        row = client.post("/api/demo").json()
        ready(client, row["id"])
        response = client.post("/api/search", json={"index_id": row["id"], "query": "zxqvkjblah"})
        assert response.json()["files"] == []


def test_public_checkout_source_links_and_errors(tmp_path, monkeypatch):
    from llm_agent_eval.indexing import CodeChunk
    def fetch(repo, ref, root):
        assert repo == "org/project" and ref == "main"
        return [CodeChunk("lib/a.py:1-1", "lib/a.py", 1, 1, "def inventory(): pass")], "a"*40, {
            "source_bytes": 25, "skipped_files": 0}
    monkeypatch.setattr("llm_agent_eval.service.fetch_sources", fetch)
    with TestClient(create_app(tmp_path)) as client:
        row = client.post("/api/indexes", json={"repository": "https://github.com/org/project",
                                               "revision": "main"}).json()
        ready(client, row["id"])
        result = client.post("/api/search", json={"index_id": row["id"], "query": "inventory"}).json()
        assert result["files"][0]["snippets"][0]["source_url"] == (
            "https://github.com/org/project/blob/" + "a"*40 + "/lib/a.py#L1-L1")


def test_git_reader_skips_symlinks_submodules_and_oversized_files(tmp_path, monkeypatch):
    def git(*args, **kwargs):
        if args[0] == "ls-tree":
            return (b"120000 blob bad 12\tsecret.py\0"
                    b"160000 commit bad -\tmodule.py\0"
                    b"100644 blob bad 500001\tlarge.py\0"
                    b"100644 blob good 20\tsafe.py\0")
        assert args == ("cat-file", "blob", "good")
        return b"def safe(): pass\n"
    monkeypatch.setattr("llm_agent_eval.repository.git", git)
    chunks, stats = read_git_sources(tmp_path, "HEAD")
    assert [c.path for c in chunks] == ["safe.py"]
    assert stats["skipped_files"] == 3


def test_published_evidence_matches_archived_artifact(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        evidence = client.get("/api/evaluation").json()
    archived = Path(__file__).parents[1] / "results/published/swebench_verified_bm25_20.json"
    assert evidence["baseline"] == json.loads(archived.read_text())


def test_interrupted_jobs_are_marked_failed(tmp_path):
    from llm_agent_eval.service import IndexService
    service = IndexService(tmp_path)
    service.save({"id": "a"*32, "status": "indexing", "created_at": "2026-01-01"})
    service.close()
    restored = IndexService(tmp_path)
    assert restored.get("a"*32)["status"] == "failed"
    restored.close()


def test_corrupt_snapshot_is_rejected_after_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        row = client.post("/api/demo").json()
        ready(client, row["id"])
    (tmp_path / f'{row["id"]}.json').write_text("{}")
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/search", json={"index_id": row["id"], "query": "inventory"})
        assert response.status_code == 409
        assert "integrity" in response.json()["detail"]


def test_dense_index_remains_lexically_searchable_when_dense_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("REPOLENS_ENABLE_DENSE", raising=False)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        row = client.post("/api/demo").json()
        row = ready(client, row["id"])
        # Simulate a snapshot previously indexed by a dense-enabled server.
        app.state.service.save({**row, "methods": ["bm25", "dense", "hybrid"]})
    with TestClient(create_app(tmp_path)) as client:
        query = {"index_id": row["id"], "query": "inventory"}
        assert client.post("/api/search", json=query).status_code == 200
        response = client.post("/api/search", json={**query, "method": "dense"})
        assert response.status_code == 409
        assert "disabled" in response.json()["detail"]


def test_public_demo_allows_search_but_requires_operator_for_mutations(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOLENS_PUBLIC_DEMO", "1")
    monkeypatch.delenv("REPOLENS_DEMO_REVISION", raising=False)
    with TestClient(create_app(tmp_path, token="operator-secret")) as client:
        assert client.get("/api/config").json()["public_demo"]
        row = client.post("/api/demo").json()
        ready(client, row["id"])
        assert client.post("/api/search", json={"index_id": row["id"], "query": "inventory"}).status_code == 200
        assert client.delete(f'/api/indexes/{row["id"]}').status_code == 403
        assert client.post("/api/indexes", json={"repository": "a/b"}).status_code == 403
        response = client.get("/")
        assert "https://huggingface.co" in response.headers["content-security-policy"]
        assert "x-frame-options" not in response.headers


def test_explanation_api_checks_configuration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/explanations", json={"index_id": "a"*32, "query": "inventory"})
        assert response.status_code == 409


def test_explanation_job_returns_validated_result(tmp_path, monkeypatch):
    from llm_agent_eval.providers import Usage
    monkeypatch.setenv("REPOLENS_ENABLE_EXPLANATIONS", "1")
    class Client:
        model_name = "fixture"
        revision = "fixture-revision"
        last_usage = Usage(10, 10, 20)
        def complete(self, prompt):
            evidence = json.loads(prompt.split("\n", 1)[1])["evidence"][0]
            return json.dumps({"findings": [{"statement": "Inspect this inventory operation.",
                                             "evidence": [{"id": evidence["id"],
                                                           "quote": evidence["text"].splitlines()[0]}]}],
                               "suggested_check": "Inspect its callers."})
    app = create_app(tmp_path)
    with TestClient(app) as client:
        app.state.explanations.client = Client()
        index = client.post("/api/demo").json()
        ready(client, index["id"])
        response = client.post("/api/explanations", json={"index_id": index["id"], "query": "inventory"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/explanations/{job_id}").json()
            if job["status"] in {"ready", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "ready", job
        assert job["result"]["model_revision"] == "fixture-revision"
