"""Sample local benchmark memory without modifying the evaluated retrieval engine."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import psutil


def gpu_memory() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=10)
    if result.returncode:
        return None
    try:
        return sum(int(line.strip()) for line in result.stdout.splitlines())
    except ValueError:
        return None


def active_benchmarks() -> list[dict]:
    rows = []
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            args = process.info["cmdline"] or []
            command = " ".join(args)
            if "scripts/run_frozen_comparison.py" not in command:
                continue
            study = "balanced_50" if "balanced-50" in command else "verified_20"
            memory = process.info["memory_info"]
            rows.append({"study": study, "pid": process.pid,
                         "rss_bytes": memory.rss,
                         "process_peak_rss_bytes": getattr(memory, "peak_wset", None)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/resource-observations.jsonl"))
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--max-hours", type=int, default=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = args.max_hours * 3600 // args.interval
    with args.output.open("a", encoding="utf-8") as target:
        for _ in range(count):
            gpu_mib = gpu_memory()
            timestamp = datetime.now(timezone.utc).isoformat()
            for row in active_benchmarks():
                target.write(json.dumps({"timestamp": timestamp, "gpu_used_mib": gpu_mib,
                                         **row}) + "\n")
            target.flush()
            if (Path("results/verified-20/comparison.json").exists()
                    and Path("results/balanced-50/comparison.json").exists()):
                break
            sleep(args.interval)


if __name__ == "__main__":
    main()
