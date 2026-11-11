"""
core/llm.py

Minimal LLM provider seam for sprint planning. Local-first:

  - "ollama"   -> local Ollama HTTP API via `requests` (already a dependency)
  - "anthropic"-> Anthropic SDK, lazy-imported, only when LLM-API-KEY is set
  - "offline"  -> deterministic template planner (no network/model)

`plan_sprint_llm` always degrades to the offline planner on any provider error,
so sprint planning works with no model installed and tests stay deterministic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Static, deterministic estimation/risk knowledge for the offline planner.
_STAGE_PLAN: dict[str, dict[str, Any]] = {
    "business": {"points": 2, "risks": ["Unclear business objective"],
                 "blockers": ["Stakeholder sign-off pending"]},
    "acquire": {"points": 5, "risks": ["Source availability/rate limits"],
                "blockers": ["Data source credentials"]},
    "profile": {"points": 2, "risks": ["High-cardinality or wide data slows profiling"],
                "blockers": []},
    "eda": {"points": 3, "risks": ["Hidden data-quality issues"], "blockers": []},
    "clean": {"points": 5, "risks": ["Aggressive cleaning drops signal"], "blockers": []},
    "feature_eng": {"points": 8, "risks": ["Leakage from look-ahead features"],
                    "blockers": []},
    "feature_select": {"points": 3, "risks": ["Dropping informative features"],
                       "blockers": []},
    "data_prep": {"points": 3, "risks": ["Train/eval scaling mismatch"], "blockers": []},
    "model": {"points": 8, "risks": ["Underfitting / no real edge"],
              "blockers": ["Compute budget"]},
    "evaluate": {"points": 3, "risks": ["Optimistic metrics from leakage"], "blockers": []},
    "explain": {"points": 3, "risks": ["Misleading global importances"], "blockers": []},
    "deploy": {"points": 5, "risks": ["Inference/training skew"],
               "blockers": ["Deployment target undefined"]},
}


# --- Provider dispatch -------------------------------------------------------

def _llm_api_key() -> str | None:
    # Reuse mira's secrets convention; fall back to common env vars.
    key = os.getenv("LLM-API-KEY") or os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        from acquire.utils.secrets import get_llm_api_key  # noqa: PLC0415
        return get_llm_api_key()
    except Exception:
        return None


def generate(prompt: str, *, provider: str = "offline", model: str | None = None,
             timeout: int = 60) -> str:
    """Return raw model text. Raises on provider failure (caller may fall back)."""
    if provider == "ollama":
        return _ollama_generate(prompt, model or DEFAULT_OLLAMA_MODEL, timeout)
    if provider == "anthropic":
        return _anthropic_generate(prompt, model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError(f"provider '{provider}' has no text generator (use offline plan)")


def _ollama_generate(prompt: str, model: str, timeout: int) -> str:
    import requests  # already a dependency

    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _anthropic_generate(prompt: str, model: str) -> str:
    key = _llm_api_key()
    if not key:
        raise RuntimeError("no LLM-API-KEY configured for anthropic provider")
    import anthropic  # lazy; optional dependency

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(getattr(b, "text", "") for b in msg.content)


# --- Sprint planning ---------------------------------------------------------

def build_prompt(context: dict[str, Any]) -> str:
    return (
        "You are an agile delivery planner. Given a project's context, produce a "
        "sprint plan as STRICT JSON with this shape:\n"
        '{"goal": str, "stories": [{"title": str, "as_a": str, "i_want": str, '
        '"so_that": str, "acceptance_criteria": [str], "points": int (Fibonacci '
        "1/2/3/5/8/13/21), \"priority\": \"low|medium|high|urgent\", \"risks\": "
        '[str], "blockers": [str]}], "risks": [str], "blockers": [str]}\n'
        "Only output JSON, no prose.\n\n"
        f"PROJECT CONTEXT:\n{json.dumps(context, indent=2)}\n"
    )


def plan_sprint_llm(context: dict[str, Any], *, provider: str = "offline",
                    model: str | None = None) -> dict[str, Any]:
    """Produce a sprint plan dict. Falls back to the offline planner on any error."""
    if provider == "offline":
        return _offline_plan(context)
    try:
        raw = generate(build_prompt(context), provider=provider, model=model)
        plan = _parse_plan(raw)
        plan["provider"] = provider
        return plan
    except Exception as exc:
        plan = _offline_plan(context)
        plan["provider"] = f"offline (fallback from {provider}: {type(exc).__name__})"
        return plan


def _parse_plan(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    plan = json.loads(match.group(0))
    plan.setdefault("stories", [])
    plan.setdefault("risks", [])
    plan.setdefault("blockers", [])
    plan.setdefault("goal", "")
    return plan


def _offline_plan(context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic plan: one story per remaining pipeline stage.

    No randomness or clock dependence — identical context -> identical plan.
    """
    remaining = context.get("remaining_stages") or []
    name = context.get("project_name", "project")
    target = context.get("target") or "the target"
    capacity = context.get("capacity")

    from core.board import STAGE_LABELS  # local import avoids a cycle at module load

    stories: list[dict[str, Any]] = []
    used = 0
    sprint_risks: list[str] = []
    sprint_blockers: list[str] = []

    for stage in remaining:
        info = _STAGE_PLAN.get(stage, {"points": 3, "risks": [], "blockers": []})
        points = info["points"]
        if capacity is not None and used + points > capacity:
            continue  # respect the capacity budget; deterministic skip
        label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
        stories.append({
            "title": f"{label}",
            "as_a": "data scientist",
            "i_want": f"to complete the {label.lower()} stage",
            "so_that": f"the pipeline progresses toward modeling {target}",
            "acceptance_criteria": [
                f"{label} stage runs successfully",
                f"{stage} appears in completed_stages",
            ],
            "points": points,
            "priority": "high" if stage in ("model", "evaluate") else "medium",
            "risks": list(info["risks"]),
            "blockers": list(info["blockers"]),
            "stage": stage,
        })
        used += points
        sprint_risks.extend(info["risks"])
        sprint_blockers.extend(info["blockers"])

    # de-dup while preserving order
    sprint_risks = list(dict.fromkeys(sprint_risks))
    sprint_blockers = list(dict.fromkeys(sprint_blockers))

    goal = (f"Advance {name} through {len(stories)} pipeline stage(s) toward a "
            f"validated model for {target}.")
    return {
        "goal": goal,
        "stories": stories,
        "risks": sprint_risks,
        "blockers": sprint_blockers,
        "total_points": used,
        "provider": "offline",
    }
