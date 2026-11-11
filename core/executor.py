"""
core/executor.py

Run a task through its bound skill, using the context assigned in core/context.
This is where the board stops *tracking* work and starts *doing* it:

  - stage skills (run_stage:<stage>) -> execute that pipeline stage
  - gated_ci                          -> run the gated CI pipeline
  - llm_agent                         -> package the context brief and (when a
                                         provider is set) hand it to an LLM run;
                                         offline/dry-run returns the brief as a
                                         ready-to-execute package (e.g. for Claude
                                         Code or a human).

Lineage is enforced from the *assigned context*: a stage/CI task whose consumed
`<stage>.output` resources are not yet available is reported `blocked` rather
than run — so context is functional, not decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import board, context, state
from core.project import Project


@dataclass
class ExecutionResult:
    status: str            # success | failed | blocked | ready | error
    skill: str = ""
    kind: str = ""
    task_id: int | None = None
    output: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("success", "ready")


def _missing_lineage(ctx: dict[str, Any]) -> list[str]:
    """Consumed `<stage>.output` resources that are not yet available."""
    resolved = ctx.get("resolved", {})
    return [
        r for r in ctx.get("needs", [])
        if r.endswith(".output") and not (resolved.get(r) or {}).get("available")
    ]


def execute_task(
    project: Project,
    task: Any,
    *,
    provider: str | None = None,
    dry_run: bool = False,
) -> ExecutionResult:
    """Execute a task via its assigned skill. Always returns (never raises)."""
    ctx = context.get_context(project, task, refresh=True)
    skill, kind = ctx["skill"], ctx["skill_kind"]
    task_id = getattr(task, "id", None)

    result = _dispatch(project, task, ctx, provider=provider, dry_run=dry_run)
    result.skill, result.kind, result.task_id = skill, kind, task_id

    try:
        state.log_execution(
            project.root / "state.db",
            task_id=task_id, skill=skill, kind=kind, status=result.status,
            provider=provider, output=result.output, error=result.error,
        )
    except Exception:
        pass
    return result


def _dispatch(project, task, ctx, *, provider, dry_run) -> ExecutionResult:
    kind = ctx["skill_kind"]

    # Lineage gate for anything that consumes upstream stage outputs.
    if kind in ("stage", "ci") and not dry_run:
        missing = _missing_lineage(ctx)
        if missing:
            return ExecutionResult(
                status="blocked",
                output=f"Missing upstream outputs: {', '.join(missing)}",
                error="lineage prerequisites not met",
            )

    if kind == "stage":
        return _run_stage(project, task, ctx, dry_run=dry_run)
    if kind == "ci":
        return _run_ci(project, dry_run=dry_run)
    return _run_agent(project, ctx, provider=provider, dry_run=dry_run)


def _run_stage(project, task, ctx, *, dry_run) -> ExecutionResult:
    stage = getattr(task, "stage", None)
    if dry_run:
        return ExecutionResult(status="ready",
                               output=f"Would run stage '{stage}' with assigned context.",
                               summary={"stage": stage, "needs": ctx["needs"]})

    from core.pipeline import Pipeline
    res = Pipeline(project).run_stage(stage)
    if res.status == "success":
        # Advance the linked card; best-effort (fires task_done trigger).
        tid = getattr(task, "id", None)
        if tid is not None:
            try:
                board.move_task(project, tid, "done")
            except Exception:
                pass
        return ExecutionResult(status="success",
                               output=f"Stage '{stage}' completed.",
                               summary=res.summary or {})
    return ExecutionResult(status="failed",
                           output=f"Stage '{stage}' failed.",
                           error=(res.error or "").strip()[:500])


def _run_ci(project, *, dry_run) -> ExecutionResult:
    from core import ci as ci_mod
    if dry_run:
        gates = ci_mod.configured_gates(project)
        return ExecutionResult(status="ready",
                               output=f"Would run gated CI ({len(gates)} gate(s)).",
                               summary={"gates": gates})
    res = ci_mod.run_ci(project)
    if res.status == "passed":
        return ExecutionResult(status="success",
                               output=f"CI passed: {', '.join(res.stages_run)}",
                               summary={"stages_run": res.stages_run})
    if res.status == "failed":
        g = res.failed_gate
        return ExecutionResult(status="failed",
                               output=f"CI failed at {g.gate if g else '?'}",
                               error=g.message if g else "gate failed")
    return ExecutionResult(status="error",
                           output=f"CI error at {res.failed_stage}",
                           error=(res.error or "").strip()[:500])


def _run_agent(project, ctx, *, provider, dry_run) -> ExecutionResult:
    from core import llm, sprint

    brief = context.render_brief(ctx)
    cfg = sprint.llm_config(project)
    provider = provider or cfg.get("provider", "offline")

    # Offline or dry-run: produce the hand-off package, don't call a model.
    if dry_run or provider == "offline":
        return ExecutionResult(status="ready", output=brief,
                               summary={"handoff": True, "provider": "offline"})

    prompt = (
        "You are an autonomous executor. Implement the task described in this "
        "brief. Respond with your plan and the concrete changes you would make.\n\n"
        + brief
    )
    try:
        text = llm.generate(prompt, provider=provider, model=cfg.get("model"))
        return ExecutionResult(status="success", output=text,
                               summary={"provider": provider})
    except Exception as exc:
        # Degrade to the hand-off package rather than failing the task.
        return ExecutionResult(
            status="ready",
            output=brief,
            summary={"handoff": True, "provider": f"offline (fallback: {type(exc).__name__})"},
        )
