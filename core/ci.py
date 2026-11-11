"""
core/ci.py

Local CI/CD for a project: a gated pipeline runner plus an in-process trigger
seam.

`run_ci` runs pipeline stages in order, evaluating the gates declared in
meta["ci"]["gates"] after each stage. A failing (blocking) gate stops the run
*before* the next stage, so e.g. `deploy` never runs on a model that missed its
metric threshold. Every gate evaluation is logged to the gate_runs table.

Triggers are in-process: `on_event` is called from board/ingest code paths and,
if the matching trigger is enabled in meta["ci"]["triggers"], kicks off a CI
run. Scheduled/cron execution is intentionally out-of-process — an external
scheduler simply invokes `mira ci run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import gates as gates_mod
from core import state
from core.pipeline import STAGE_ORDER, Pipeline
from core.project import Project


@dataclass
class CIResult:
    status: str  # passed | failed | error
    stages_run: list[str] = field(default_factory=list)
    gate_results: list[gates_mod.GateResult] = field(default_factory=list)
    failed_gate: gates_mod.GateResult | None = None
    failed_stage: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "passed"


def ci_config(project: Project) -> dict[str, Any]:
    """Self-healing meta["ci"] config."""
    ci = project.meta.setdefault("ci", {})
    changed = False
    if "gates" not in ci:
        ci["gates"] = []
        changed = True
    if "triggers" not in ci:
        # Which in-process events auto-launch a CI run.
        ci["triggers"] = {"task_done": False, "new_data": False}
        changed = True
    if changed:
        project.save_meta()
    return ci


def configured_gates(project: Project) -> list[dict[str, Any]]:
    return ci_config(project).get("gates", [])


def run_ci(project: Project, *, through: str | None = None,
           from_stage: str | None = None) -> CIResult:
    """Run the gated pipeline. Stops at the first blocking gate failure."""
    db = project.root / "state.db"
    gates = configured_gates(project)
    pipe = Pipeline(project)

    start = STAGE_ORDER.index(from_stage) if from_stage in STAGE_ORDER else 0
    result = CIResult(status="passed")

    for name in STAGE_ORDER[start:]:
        stage_res = pipe.run_stage(name)
        result.stages_run.append(name)

        if stage_res.status == "error":
            result.status = "error"
            result.failed_stage = name
            result.error = stage_res.error
            return result

        stage_gates = gates_mod.evaluate_gates(name, stage_res.summary or {}, gates)
        for gr in stage_gates:
            result.gate_results.append(gr)
            state.log_gate_run(
                db, stage=gr.stage, gate=gr.gate, status=gr.status,
                observed=gr.observed, expected=gr.expected, message=gr.message,
            )
            if gr.blocking:
                result.status = "failed"
                result.failed_gate = gr
                result.failed_stage = name
                return result

        if through and name == through:
            break

    return result


# --- In-process trigger seam -------------------------------------------------

def on_event(project: Project, event: str, payload: dict[str, Any] | None = None) -> None:
    """Handle an in-process event (e.g. 'task_done', 'new_data').

    Logs the event and, if the matching trigger is enabled, launches a CI run.
    Best-effort: never raises into the caller (board moves, ingest, ...).
    """
    db = project.root / "state.db"

    # Resolve-on-trigger: refresh any context snapshots the event invalidated.
    # Best-effort and independent of whether a CI run is also enabled.
    if event in ("new_data", "stage_rerun"):
        try:
            from core import context as context_mod
            context_mod.refresh_stale(project)
        except Exception:
            pass

    triggers = ci_config(project).get("triggers", {})
    enabled = bool(triggers.get(event))
    action = "ci_run" if enabled else "noop"
    outcome = "skipped"

    try:
        if enabled:
            res = run_ci(project)
            outcome = res.status
    except Exception as exc:  # pragma: no cover - defensive
        outcome = f"error: {exc}"
    finally:
        try:
            state.log_trigger_event(
                db, event=event, payload=payload or {}, action=action, result=outcome,
            )
        except Exception:
            pass
