"""Complete both frozen comparisons and prepare verified reports for review."""
from __future__ import annotations

import shutil
import subprocess
import sys
import argparse
import os
from pathlib import Path
from time import sleep


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def study(directory: str, tasks: str, manifest: str, title: str, label: str) -> None:
    run("scripts/run_frozen_comparison.py", tasks, "--manifest", manifest,
        "--output-dir", directory)
    output = Path(directory)
    if not (output / "comparison.json").is_file():
        raise RuntimeError(f"The {title} comparison was not completed")
    report = output / "report.json"
    run("scripts/summarize_study.py", directory, "--name", title, "--output", str(report))
    published = Path("results/published")
    published.mkdir(parents=True, exist_ok=True)
    for method in ("bm25", "dense", "hybrid"):
        shutil.copy2(output / f"{method}.json", published / f"{label}_{method}.json")
    shutil.copy2(report, published / f"{label}_report.json")
    evidence = Path("src/llm_agent_eval/evidence")
    shutil.copy2(report, evidence / f"study_{label}.json")
    print(f"Validated report: {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int,
                        help="Wait for an existing benchmark runner before resuming checkpoints")
    args = parser.parse_args()
    if args.wait_pid:
        while not Path("results/verified-20/comparison.json").exists():
            try:
                os.kill(args.wait_pid, 0)
            except OSError:
                break
            sleep(30)
    study("results/verified-20", ".benchmark/study.jsonl",
          "configs/verified_20_manifest.json", "Frozen 20-task comparison", "verified_20")
    study("results/balanced-50", ".benchmark/balanced-50.jsonl",
          "configs/verified_balanced_50_manifest.json", "Repository-balanced 50-task comparison",
          "balanced_50")


if __name__ == "__main__":
    main()
