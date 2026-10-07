from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".php", ".scala", ".kt",
}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


def iter_source_files(root: Path, extensions: set[str] | None = None):
    extensions = extensions or DEFAULT_EXTENSIONS
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def chunk_repository(
    root: str | Path,
    *,
    lines_per_chunk: int = 80,
    overlap: int = 20,
) -> list[CodeChunk]:
    if lines_per_chunk <= 0 or overlap < 0 or overlap >= lines_per_chunk:
        raise ValueError("Require lines_per_chunk > overlap >= 0")

    root = Path(root).resolve()
    chunks: list[CodeChunk] = []
    step = lines_per_chunk - overlap

    for path in iter_source_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        relative = path.relative_to(root).as_posix()
        for start in range(0, len(lines), step):
            block = lines[start : start + lines_per_chunk]
            if not block:
                break
            start_line = start + 1
            end_line = start + len(block)
            chunk_id = f"{relative}:{start_line}-{end_line}"
            chunks.append(CodeChunk(chunk_id, relative, start_line, end_line, "\n".join(block)))
            if end_line == len(lines):
                break

    return chunks
