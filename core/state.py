"""
core/state.py

SQLite-backed run log for stage executions (per project).
Minimal implementation for Phase 1; expanded in later phases.
"""

import datetime
import json
import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any, Optional


def init_db(path: Path) -> None:
    """Initialize the stage_runs table if it does not exist."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stage_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL,
            run_at TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_ms INTEGER,
            artifacts_json TEXT,
            summary_json TEXT,
            error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_stage_runs_stage ON stage_runs(stage);
        CREATE INDEX IF NOT EXISTS idx_stage_runs_run_at ON stage_runs(run_at);
    """)
    conn.commit()
    conn.close()
    # Board / sprint / CI tables are additive and live alongside stage_runs.
    ensure_board_schema(path)


# --- Board / sprint / CI schema (additive; safe to call repeatedly) ----------

_BOARD_SCHEMA = """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        col TEXT NOT NULL DEFAULT 'backlog',
        priority TEXT DEFAULT 'medium',
        points INTEGER,
        order_index REAL NOT NULL DEFAULT 0,
        sprint_id INTEGER,
        blockers_json TEXT,
        risks_json TEXT,
        assignee TEXT,
        labels_json TEXT,
        stage TEXT,
        created_at TEXT,
        updated_at TEXT,
        status_changed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_col ON tasks(col);
    CREATE INDEX IF NOT EXISTS idx_tasks_sprint ON tasks(sprint_id);

    CREATE TABLE IF NOT EXISTS sprints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        goal TEXT,
        state TEXT DEFAULT 'planned',
        capacity_points INTEGER,
        plan_json TEXT,
        provider TEXT,
        created_at TEXT,
        started_at TEXT,
        closed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS gate_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT,
        stage TEXT,
        gate TEXT,
        status TEXT,
        observed_json TEXT,
        expected_json TEXT,
        message TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_gate_runs_stage ON gate_runs(stage);

    CREATE TABLE IF NOT EXISTS trigger_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fired_at TEXT,
        event TEXT,
        payload_json TEXT,
        action TEXT,
        result TEXT
    );

    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT,
        task_id INTEGER,
        skill TEXT,
        kind TEXT,
        status TEXT,
        provider TEXT,
        output TEXT,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_executions_task ON executions(task_id);
"""


# Columns added after the initial tasks table shipped. Added via ALTER TABLE so
# projects whose tasks table predates them self-heal (additive, idempotent).
_TASK_ADDED_COLUMNS = {
    "context_spec_json": "TEXT",
    "context_snapshot_json": "TEXT",
    "context_fingerprint": "TEXT",
    "context_assigned_at": "TEXT",
}


def _migrate_columns(conn: sqlite3.Connection, table: str,
                     columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col, decl in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def ensure_board_schema(path: Path) -> None:
    """Create the board/sprint/CI tables if missing.

    `init_db` only runs at project create time, so call this lazily from board
    code paths to self-heal projects that predate the board feature.
    """
    conn = sqlite3.connect(path)
    conn.executescript(_BOARD_SCHEMA)
    _migrate_columns(conn, "tasks", _TASK_ADDED_COLUMNS)
    conn.commit()
    conn.close()


def log_run(
    path: Path,
    *,
    stage: str,
    settings: dict[str, Any],
    status: str,
    duration_ms: int,
    artifacts: Optional[list[str]] = None,
    summary: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Append a stage run record to the project's state.db."""
    artifacts = artifacts or []
    summary = summary or {}

    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO stage_runs
            (stage, run_at, settings_json, status, duration_ms, artifacts_json, summary_json, error_message)
        VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
        """,
        (
            stage,
            json.dumps(settings),
            status,
            duration_ms,
            json.dumps(artifacts),
            json.dumps(summary),
            error,
        ),
    )
    conn.commit()
    conn.close()


def get_last_run(path: Path, stage: str) -> Optional[dict[str, Any]]:
    """Return the most recent run record for a given stage, or None."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_runs WHERE stage = ? ORDER BY run_at DESC LIMIT 1",
        (stage,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def get_all_runs(path: Path) -> list[dict[str, Any]]:
    """Return all run records for the project (newest first)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM stage_runs ORDER BY run_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Task CRUD ---------------------------------------------------------------

_TASK_COLUMNS = (
    "title", "description", "col", "priority", "points", "order_index",
    "sprint_id", "blockers_json", "risks_json", "assignee", "labels_json",
    "stage", "created_at", "updated_at", "status_changed_at",
)


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat() + "Z"


def add_task(path: Path, fields: dict[str, Any]) -> int:
    """Insert a task row. `fields` keys are db column names. Returns new id."""
    ensure_board_schema(path)
    data = {k: fields.get(k) for k in _TASK_COLUMNS}
    now = _now()
    data["created_at"] = data.get("created_at") or now
    data["updated_at"] = now
    data["status_changed_at"] = data.get("status_changed_at") or now
    for json_col in ("blockers_json", "risks_json", "labels_json"):
        if isinstance(data.get(json_col), (list, dict)):
            data[json_col] = json.dumps(data[json_col])
    if data.get("order_index") is None:
        data["order_index"] = _next_order_index(path, data.get("col") or "backlog")

    cols = ", ".join(_TASK_COLUMNS)
    placeholders = ", ".join("?" for _ in _TASK_COLUMNS)
    conn = sqlite3.connect(path)
    cur = conn.execute(
        f"INSERT INTO tasks ({cols}) VALUES ({placeholders})",
        tuple(data[c] for c in _TASK_COLUMNS),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return int(new_id)


def _next_order_index(path: Path, col: str) -> float:
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT MAX(order_index) FROM tasks WHERE col = ?", (col,)
    ).fetchone()
    conn.close()
    top = row[0] if row and row[0] is not None else 0.0
    return float(top) + 1.0


def update_task(path: Path, task_id: int, fields: dict[str, Any]) -> None:
    """Update arbitrary columns on a task. JSON-list fields auto-encoded."""
    ensure_board_schema(path)
    allowed = {k: v for k, v in fields.items() if k in _TASK_COLUMNS}
    if not allowed:
        return
    for json_col in ("blockers_json", "risks_json", "labels_json"):
        if isinstance(allowed.get(json_col), (list, dict)):
            allowed[json_col] = json.dumps(allowed[json_col])
    allowed["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in allowed)
    conn = sqlite3.connect(path)
    conn.execute(
        f"UPDATE tasks SET {sets} WHERE id = ?",
        (*allowed.values(), task_id),
    )
    conn.commit()
    conn.close()


def move_task(path: Path, task_id: int, col: str, order_index: float | None = None) -> None:
    """Move a task to a different column (and optionally a new order slot)."""
    ensure_board_schema(path)
    if order_index is None:
        order_index = _next_order_index(path, col)
    now = _now()
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE tasks SET col = ?, order_index = ?, status_changed_at = ?, updated_at = ? "
        "WHERE id = ?",
        (col, order_index, now, now, task_id),
    )
    conn.commit()
    conn.close()


def delete_task(path: Path, task_id: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def get_task(path: Path, task_id: int) -> Optional[dict[str, Any]]:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_task_context(
    path: Path,
    task_id: int,
    *,
    spec: Any,
    snapshot: Any,
    fingerprint: str,
) -> None:
    """Persist a task's assigned context (spec + frozen snapshot + fingerprint)."""
    ensure_board_schema(path)
    now = _now()
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE tasks SET context_spec_json = ?, context_snapshot_json = ?, "
        "context_fingerprint = ?, context_assigned_at = ?, updated_at = ? "
        "WHERE id = ?",
        (json.dumps(spec), json.dumps(snapshot), fingerprint, now, now, task_id),
    )
    conn.commit()
    conn.close()


def list_tasks(path: Path, sprint_id: Optional[int] = None) -> list[dict[str, Any]]:
    """Return stored tasks ordered by column then order_index."""
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if sprint_id is None:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY col, order_index"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE sprint_id = ? ORDER BY col, order_index",
            (sprint_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Sprints -----------------------------------------------------------------

def add_sprint(path: Path, fields: dict[str, Any]) -> int:
    ensure_board_schema(path)
    cols = ("name", "goal", "state", "capacity_points", "plan_json", "provider",
            "created_at", "started_at", "closed_at")
    data = {k: fields.get(k) for k in cols}
    data["created_at"] = data.get("created_at") or _now()
    if isinstance(data.get("plan_json"), (list, dict)):
        data["plan_json"] = json.dumps(data["plan_json"])
    placeholders = ", ".join("?" for _ in cols)
    conn = sqlite3.connect(path)
    cur = conn.execute(
        f"INSERT INTO sprints ({', '.join(cols)}) VALUES ({placeholders})",
        tuple(data[c] for c in cols),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return int(new_id)


def list_sprints(path: Path) -> list[dict[str, Any]]:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sprints ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Gate runs & trigger events ----------------------------------------------

def log_gate_run(
    path: Path,
    *,
    stage: str,
    gate: str,
    status: str,
    observed: Any = None,
    expected: Any = None,
    message: Optional[str] = None,
) -> None:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO gate_runs (run_at, stage, gate, status, observed_json, expected_json, message) "
        "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (stage, gate, status, json.dumps(observed), json.dumps(expected), message),
    )
    conn.commit()
    conn.close()


def get_gate_runs(path: Path) -> list[dict[str, Any]]:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM gate_runs ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_execution(
    path: Path,
    *,
    task_id: Optional[int],
    skill: str,
    kind: str,
    status: str,
    provider: Optional[str] = None,
    output: Optional[str] = None,
    error: Optional[str] = None,
) -> int:
    """Record a task execution for provenance. Returns the new row id."""
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "INSERT INTO executions (run_at, task_id, skill, kind, status, provider, output, error) "
        "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
        (task_id, skill, kind, status, provider,
         (output or "")[:4000], error),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return int(new_id)


def get_executions(path: Path, task_id: Optional[int] = None) -> list[dict[str, Any]]:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if task_id is None:
        rows = conn.execute("SELECT * FROM executions ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM executions WHERE task_id = ? ORDER BY id DESC", (task_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_trigger_event(
    path: Path,
    *,
    event: str,
    payload: Any = None,
    action: Optional[str] = None,
    result: Optional[str] = None,
) -> None:
    ensure_board_schema(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO trigger_events (fired_at, event, payload_json, action, result) "
        "VALUES (datetime('now'), ?, ?, ?, ?)",
        (event, json.dumps(payload), action, result),
    )
    conn.commit()
    conn.close()
