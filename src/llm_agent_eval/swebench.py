from __future__ import annotations

import re
from dataclasses import dataclass

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def gold_files_from_patch(patch: str) -> frozenset[str]:
    """Extract changed file paths from a unified git patch for evaluation labels only."""
    files: set[str] = set()
    for before, after in DIFF_FILE_RE.findall(patch):
        if before != "/dev/null":
            files.add(before)
        if after != "/dev/null":
            files.add(after)
    return frozenset(files)


@dataclass(frozen=True)
class SWEBenchRecord:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str

    @property
    def gold_files(self) -> frozenset[str]:
        return gold_files_from_patch(self.patch)


def from_mapping(record: dict) -> SWEBenchRecord:
    required = {"instance_id", "repo", "base_commit", "problem_statement", "patch"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing SWE-bench fields: {sorted(missing)}")
    return SWEBenchRecord(
        instance_id=str(record["instance_id"]),
        repo=str(record["repo"]),
        base_commit=str(record["base_commit"]),
        problem_statement=str(record["problem_statement"]),
        patch=str(record["patch"]),
    )
