"""
core/git_sync.py

Read-only git awareness for a project ("the app syncs with mira"). Each project
may point at its own repo via meta["git"]; this module shells out to `git` to
read branch / dirty / ahead-behind / last-commit state. Everything degrades
gracefully when git is missing or the path is not a repository.

Push / remote-CI integration is intentionally deferred (see plan Phase 6).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.project import Project


def git_config(project: Project) -> dict[str, Any]:
    """Self-healing meta["git"] config. repo_path defaults to the project root."""
    cfg = project.meta.setdefault("git", {})
    if "repo_path" not in cfg:
        cfg["repo_path"] = str(project.root.resolve())
        project.save_meta()
    return cfg


def _run_git(repo: Path, *args: str, timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        # git binary missing, timeout, or other OS error.
        return 127, ""


def _is_repo(repo: Path) -> bool:
    code, out = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    return code == 0 and out == "true"


def git_status(project: Project) -> dict[str, Any]:
    """Return a structured, never-raising snapshot of the project's git state.

    Keys: is_repo, repo_path, branch, dirty, ahead, behind, last_commit, error.
    """
    cfg = git_config(project)
    repo = Path(cfg.get("repo_path") or project.root)
    result: dict[str, Any] = {
        "is_repo": False,
        "repo_path": str(repo),
        "branch": None,
        "dirty": None,
        "ahead": 0,
        "behind": 0,
        "last_commit": None,
        "error": None,
    }

    if not repo.exists():
        result["error"] = "repo path does not exist"
        return result
    if not _is_repo(repo):
        result["error"] = "not a git repository"
        return result

    result["is_repo"] = True

    _, branch = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    result["branch"] = branch or None

    code, status = _run_git(repo, "status", "--porcelain")
    if code == 0:
        result["dirty"] = bool(status.strip())

    # ahead/behind vs upstream, when an upstream is configured.
    code, counts = _run_git(
        repo, "rev-list", "--left-right", "--count", "@{upstream}...HEAD"
    )
    if code == 0 and counts:
        parts = counts.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            result["behind"], result["ahead"] = int(parts[0]), int(parts[1])

    code, commit = _run_git(
        repo, "log", "-1", "--pretty=format:%h\x1f%s\x1f%an\x1f%cI"
    )
    if code == 0 and commit:
        sha, _, rest = commit.partition("\x1f")
        subject, _, rest = rest.partition("\x1f")
        author, _, when = rest.partition("\x1f")
        result["last_commit"] = {
            "sha": sha, "subject": subject, "author": author, "date": when,
        }

    return result


def status_line(status: dict[str, Any]) -> str:
    """One-line human summary for CLI/UI headers."""
    if not status.get("is_repo"):
        return f"no git repo ({status.get('error', 'unknown')})"
    branch = status.get("branch") or "?"
    bits = [f"branch {branch}"]
    if status.get("dirty"):
        bits.append("dirty")
    else:
        bits.append("clean")
    if status.get("ahead"):
        bits.append(f"↑{status['ahead']}")
    if status.get("behind"):
        bits.append(f"↓{status['behind']}")
    lc = status.get("last_commit")
    if lc:
        bits.append(f"@ {lc['sha']} {lc['subject'][:50]}")
    return "  ".join(bits)


# --- Remote CI status (read-only, via the gh CLI) ----------------------------

# Fields requested from `gh run list --json`.
_GH_FIELDS = (
    "databaseId,workflowName,status,conclusion,headBranch,displayTitle,"
    "createdAt,url,event"
)


def gh_available() -> bool:
    """True if the GitHub CLI is on PATH."""
    return shutil.which("gh") is not None


def _gh_run_list_json(repo: Path, limit: int) -> str:
    """Return raw stdout of `gh run list --json ...` (isolated for testing)."""
    proc = subprocess.run(
        ["gh", "run", "list", "--limit", str(limit), "--json", _GH_FIELDS],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "gh run list failed").strip())
    return proc.stdout or "[]"


def remote_ci_status(project: Project, *, limit: int = 5) -> dict[str, Any]:
    """Read recent GitHub Actions runs for the project's repo via `gh`.

    Never raises. Returns: available, error, runs (list of normalized dicts).
    """
    cfg = git_config(project)
    repo = Path(cfg.get("repo_path") or project.root)
    result: dict[str, Any] = {"available": False, "error": None, "runs": []}

    if not gh_available():
        result["error"] = "gh CLI not installed"
        return result
    if not repo.exists() or not _is_repo(repo):
        result["error"] = "not a git repository"
        return result

    try:
        raw = _gh_run_list_json(repo, limit)
        runs = json.loads(raw) if raw.strip() else []
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        # gh not authed, no remote/workflows, timeout, bad JSON, etc.
        result["error"] = str(exc).splitlines()[0] if str(exc) else "gh error"
        return result

    result["available"] = True
    result["runs"] = [
        {
            "id": r.get("databaseId"),
            "workflow": r.get("workflowName"),
            "status": r.get("status"),          # queued|in_progress|completed
            "conclusion": r.get("conclusion"),  # success|failure|cancelled|...
            "branch": r.get("headBranch"),
            "title": r.get("displayTitle"),
            "event": r.get("event"),
            "created_at": r.get("createdAt"),
            "url": r.get("url"),
        }
        for r in runs
    ]
    return result


_CI_ICON = {
    "success": "🟢", "failure": "🔴", "cancelled": "⚪", "skipped": "⚪",
    "startup_failure": "🔴", "timed_out": "🔴", "action_required": "🟠",
    "neutral": "⚪", "stale": "⚪",
}


def ci_run_icon(run: dict[str, Any]) -> str:
    if run.get("status") != "completed":
        return "🟡"  # queued / in_progress
    return _CI_ICON.get(run.get("conclusion"), "⚪")


def ci_status_line(status: dict[str, Any]) -> str:
    """One-line summary of the most recent remote CI run."""
    if not status.get("available"):
        return f"CI: {status.get('error', 'unavailable')}"
    runs = status.get("runs") or []
    if not runs:
        return "CI: no runs found"
    r = runs[0]
    state = r.get("conclusion") or r.get("status") or "?"
    return f"CI: {ci_run_icon(r)} {state} · {r.get('workflow', '?')} @ {r.get('branch', '?')}"
