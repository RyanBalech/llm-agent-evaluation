"""Build a static, interactive deployment from committed public source snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

from llm_agent_eval.repository import split_source


def build(output: Path):
    package = Path("src/llm_agent_eval")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package / "static", output / "static", dirs_exist_ok=True)
    html = (package / "static/index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/', 'href="./static/').replace('src="/static/', 'src="./static/')
    html = html.replace('href="/"', 'href="./"').replace('href="/docs"',
        'href="https://github.com/RyanBalech/llm-agent-evaluation/blob/feat/repolens-product/docs/PRODUCT.md"')
    html = html.replace("API reference ↗", "Product guide ↗")
    html = html.replace('<script src="./static/app.js" defer></script>',
        '<script src="./static/pages.js" defer></script>\n    <script src="./static/app.js" defer></script>')
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / ".nojekyll").touch()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for repository, root in [("demo", package / "demo"), ("RyanBalech/llm-agent-evaluation", Path("."))]:
        paths = sorted(root.glob("*.py")) if repository == "demo" else [
            Path(p) for p in subprocess.check_output(["git", "ls-files", "*.py"], text=True).splitlines()]
        chunks = []
        for path in paths:
            text = (path.read_text(encoding="utf-8") if repository == "demo" else
                    subprocess.check_output(["git", "show", f"{commit}:{path.as_posix()}"], text=True,
                                            encoding="utf-8"))
            chunks.extend(split_source(path.relative_to(root).as_posix(), text))
        serialized = [asdict(c) for c in chunks]
        digest = hashlib.sha256(json.dumps(serialized).encode()).hexdigest()
        row = {"id": hashlib.md5(repository.encode()).hexdigest(), "repository": repository,
               "commit": f"demo-{digest[:12]}" if repository == "demo" else commit,
               "status": "ready", "files": len(paths), "chunks": len(chunks),
               "methods": ["bm25"], "index_sha256": digest}
        rows.append(row)
        (output / f'{row["id"]}.json').write_text(json.dumps({"index": row, "chunks": serialized}),
                                                 encoding="utf-8")
    (output / "indexes.json").write_text(json.dumps(rows), encoding="utf-8")
    baseline = json.loads((package / "evidence/baseline.json").read_text())
    studies = [json.loads(p.read_text()) for p in sorted((package / "evidence").glob("study_*.json"))]
    evaluation = {"baseline": baseline, "studies": studies,
                  "label": "Archived 20-task SWE-bench Verified experiment",
                  "source": "https://github.com/RyanBalech/llm-agent-evaluation/blob/main/results/published/swebench_verified_bm25_20.json",
                  "limitations": ["The archived baseline has 12 Django tasks out of 20.",
                                   "File localization does not measure successful bug fixes.",
                                   "Benchmark timings do not describe browser performance."]}
    (output / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    modules = ["__init__", "browser_runtime", "indexing", "retrieval", "evaluation", "metrics",
               "datasets", "explanations", "providers"]
    with zipfile.ZipFile(output / "engine.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in modules:
            archive.write(package / f"{name}.py", f"llm_agent_eval/{name}.py")
    (output / "deployment.json").write_text(json.dumps({"source_commit": commit,
        "search_runtime": "pyodide-0.27.7", "hosting": "static", "indexes": rows}), encoding="utf-8")
    print(f"Built {output}: {len(rows)} source snapshots at {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/pages"))
    build(parser.parse_args().output)
