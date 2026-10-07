from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalizationTask:
    task_id: str
    issue: str
    gold_files: frozenset[str]


def load_jsonl(path: str | Path) -> list[LocalizationTask]:
    tasks: list[LocalizationTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            gold = frozenset(record["gold_files"])
            if not gold:
                raise ValueError(f"Task on line {line_number} has no gold files")
            tasks.append(LocalizationTask(str(record["task_id"]), record["issue"], gold))
    return tasks
