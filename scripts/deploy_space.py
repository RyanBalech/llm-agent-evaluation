"""Publish the checked-out application to a Hugging Face Docker Space on CPU Basic.

Authenticate locally with `hf auth login` first. Uploads an explicit source allowlist;
runtime data, benchmark task text, credentials, and Git metadata are never uploaded.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", help="owner/name; defaults to the authenticated user's repolens")
    args = parser.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repo = args.space or f"{user}/repolens"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit source changes before deployment so the published revision is exact")
    # Do not change an existing Space's visibility or hardware allocation implicitly.
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    with TemporaryDirectory(prefix="repolens-space-") as tmp:
        stage = Path(tmp)
        shutil.copytree("src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                                            "*.egg-info"))
        shutil.copy2("pyproject.toml", stage / "pyproject.toml")
        (stage / "README.md").write_text(
            "---\ntitle: RepoLens\nemoji: 🔎\ncolorFrom: green\ncolorTo: blue\n"
            "sdk: docker\napp_port: 7860\npinned: false\n---\n\n"
            "# RepoLens\n\nSource-linked repository issue localization and retrieval evaluation.\n\n"
            "Explore prepared repositories, inspect source evidence, and generate cited explanations "
            "with a local model. No inference API is used.\n\n"
            "This Space uses ephemeral storage; prepared indexes are rebuilt after a restart.\n\n"
            f"Source revision: `{commit}`.\n\n"
            "[Source and documentation](https://github.com/RyanBalech/llm-agent-evaluation).\n",
            encoding="utf-8")
        (stage / "Dockerfile").write_text(
            "FROM python:3.12-slim\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates "
            "&& rm -rf /var/lib/apt/lists/*\n"
            "RUN useradd --create-home --uid 1000 app\nWORKDIR /app\n"
            "COPY pyproject.toml README.md ./\nCOPY src ./src\n"
            "RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu "
            "&& pip install --no-cache-dir '.[app,ml]'\n"
            "RUN mkdir -p /app/data /app/models && chown app:app /app/data /app/models\n"
            "USER app\nENV HF_HOME=/app/models REPOLENS_DATA_DIR=/app/data "
            "REPOLENS_PUBLIC_DEMO=1 REPOLENS_ENABLE_EXPLANATIONS=1 "
            "REPOLENS_EXPLAIN_DEVICE=cpu REPOLENS_HOSTS=*.hf.space "
            f"REPOLENS_DEMO_REVISION={commit} "
            "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1\n"
            "RUN python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', "
            "revision='2e1fd397ee46e1388853d2af2c993145b0f1098a', "
            "allow_patterns=['*.json','*.safetensors','*.txt'])\"\n"
            "EXPOSE 7860\nCMD [\"repolens\",\"--host\",\"0.0.0.0\",\"--port\",\"7860\"]\n",
            encoding="utf-8")
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=stage,
                          commit_message=f"Deploy RepoLens {commit[:12]}")
    print(f"Published source: https://huggingface.co/spaces/{repo}")
    print("Check the Space runtime status before reporting the application as live.")


if __name__ == "__main__":
    main()
