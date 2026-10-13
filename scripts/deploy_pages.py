"""Build and publish the static demo using the existing GitHub authentication."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from build_pages import build

REPOSITORY = "RyanBalech/llm-agent-evaluation"


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def main():
    if run("git", "status", "--porcelain"):
        raise RuntimeError("Commit source changes first so the deployment has an exact revision")
    commit = run("git", "rev-parse", "HEAD")
    author = run("git", "config", "user.name")
    email = run("git", "config", "user.email")
    remote = f"https://github.com/{REPOSITORY}.git"
    exists = bool(run("git", "ls-remote", "--heads", remote, "gh-pages"))
    with TemporaryDirectory(prefix="repolens-pages-") as tmp:
        stage = Path(tmp)
        run("git", "init", "-b", "gh-pages", cwd=stage)
        run("git", "remote", "add", "origin", remote, cwd=stage)
        if exists:
            run("git", "fetch", "--depth=1", "origin", "gh-pages", cwd=stage)
            run("git", "checkout", "-B", "gh-pages", "FETCH_HEAD", cwd=stage)
        build(stage)
        run("git", "config", "user.name", author, cwd=stage)
        run("git", "config", "user.email", email, cwd=stage)
        run("git", "add", ".", cwd=stage)
        if run("git", "status", "--porcelain", cwd=stage):
            run("git", "commit", "-m", f"Publish RepoLens {commit[:12]}", cwd=stage)
            run("git", "push", "origin", "HEAD:gh-pages", cwd=stage)
    check = subprocess.run(["gh", "api", f"repos/{REPOSITORY}/pages"], capture_output=True,
                           text=True, check=False)
    if check.returncode == 0:
        current = json.loads(check.stdout)
        source = current.get("source", {})
        if source.get("branch") != "gh-pages" or source.get("path") != "/":
            raise RuntimeError("An existing Pages configuration uses another source; inspect it before changing")
    elif "404" in check.stderr or '"status":"404"' in check.stdout:
        run("gh", "api", "--method", "POST", f"repos/{REPOSITORY}/pages",
            "-f", "source[branch]=gh-pages", "-f", "source[path]=/")
    else:
        raise RuntimeError(f"Cannot inspect Pages configuration: {check.stderr}")
    print("Published source; verify the Pages build and browser before announcing the live URL:")
    print("https://ryanbalech.github.io/llm-agent-evaluation/")


if __name__ == "__main__":
    main()
