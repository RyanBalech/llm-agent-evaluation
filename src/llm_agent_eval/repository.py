"""Read public GitHub source blobs without checking out or executing repository code."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .indexing import DEFAULT_EXTENSIONS, SKIP_DIRS, CodeChunk

REPO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}")
REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,150}")
MAX_SOURCE_BYTES = 20_000_000
MAX_FILE_BYTES = 500_000
MAX_FILES = 4000
MAX_CHUNKS = 20_000


def normalize_repository(value: str) -> str:
    value = value.strip().removeprefix("https://github.com/").removesuffix("/")
    value = value.removesuffix(".git")
    if not REPO_PATTERN.fullmatch(value) or value.split("/")[1] in {".", ".."}:
        raise ValueError("Use owner/repository or an HTTPS github.com repository URL")
    return value


def validate_ref(value: str) -> str:
    if value and (not REF_PATTERN.fullmatch(value) or ".." in value or "//" in value):
        raise ValueError("Revision must be a branch, tag, or commit SHA")
    return value


def git(*args: str, cwd: Path, timeout: int = 120) -> bytes:
    env = os.environ.copy()
    # Do not inherit credentials, URL rewrites, external filters, or Git directory overrides.
    for key in list(env):
        if key.startswith("GIT_"):
            del env[key]
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    try:
        return subprocess.run(
            ["git", "-c", "credential.helper=", "-c", "http.followRedirects=false",
             "-c", "protocol.file.allow=never", *args], cwd=cwd, env=env,
            check=True, capture_output=True, timeout=timeout,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Repository operation timed out; try a smaller repository") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Git could not read this public repository or revision") from exc


def split_source(path: str, text: str) -> list[CodeChunk]:
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), 60):
        end = min(start + 80, len(lines))
        chunks.append(CodeChunk(f"{path}:{start + 1}-{end}", path, start + 1, end,
                                "\n".join(lines[start:end])))
        if end == len(lines):
            break
    return chunks


def read_workspace_sources(root: Path) -> tuple[list[CodeChunk], dict]:
    """Snapshot supported source files from a configured local workspace."""
    root = root.resolve(strict=True)
    chunks = []
    source_bytes = 0
    files = 0
    skipped = 0
    excluded = SKIP_DIRS | {".repolens", ".repolens-mcp", ".benchmark", "artifacts"}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() not in DEFAULT_EXTENSIONS or path.is_symlink():
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                skipped += 1
                continue
            files += 1
            source_bytes += size
            if files > MAX_FILES or source_bytes > MAX_SOURCE_BYTES:
                raise ValueError("Workspace exceeds the limit: 4,000 source files / 20 MB")
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                skipped += 1
                continue
            if "\0" in content:
                skipped += 1
                continue
            chunks.extend(split_source(path.relative_to(root).as_posix(), content))
            if len(chunks) > MAX_CHUNKS:
                raise ValueError("Workspace exceeds the limit of 20,000 chunks")
    if not chunks:
        raise ValueError("No supported, UTF-8 source files found in the workspace")
    return chunks, {"source_bytes": source_bytes, "skipped_files": skipped}


def read_git_sources(root: Path, revision: str) -> tuple[list[CodeChunk], dict]:
    entries = git("ls-tree", "-r", "-l", "-z", revision, cwd=root).split(b"\0")
    blobs = []
    skipped = 0
    total = 0
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid, size = metadata.split()
        path = raw_path.decode("utf-8", errors="replace")
        p = Path(path)
        if p.suffix.lower() not in DEFAULT_EXTENSIONS or any(x in SKIP_DIRS for x in p.parts):
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            skipped += 1
            continue
        if int(size) > MAX_FILE_BYTES:
            skipped += 1
            continue
        total += int(size)
        blobs.append((path, oid.decode()))
    if len(blobs) > MAX_FILES or total > MAX_SOURCE_BYTES:
        raise ValueError("Repository exceeds the demo limit: 4,000 source files / 20 MB of source")
    chunks = []
    for path, oid in sorted(blobs):
        raw = git("cat-file", "blob", oid, cwd=root)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped += 1
            continue
        if "\0" in text:
            skipped += 1
            continue
        chunks.extend(split_source(path, text))
        if len(chunks) > MAX_CHUNKS:
            raise ValueError("Repository exceeds the demo limit of 20,000 chunks")
    if not chunks:
        raise ValueError("No supported, UTF-8 source files found")
    return chunks, {"source_bytes": total, "skipped_files": skipped}


def fetch_sources(repo: str, ref: str, root: Path) -> tuple[list[CodeChunk], str, dict]:
    repo, ref = normalize_repository(repo), validate_ref(ref)
    root.mkdir(parents=True)
    git("init", "--bare", ".", cwd=root)
    git("fetch", "--depth=1", "--no-tags", f"https://github.com/{repo}.git",
        ref or "HEAD", cwd=root)
    commit = git("rev-parse", "FETCH_HEAD^{commit}", cwd=root).decode().strip()
    chunks, stats = read_git_sources(root, commit)
    return chunks, commit, stats
