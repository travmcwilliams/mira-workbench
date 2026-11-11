"""
core/skills.py

Skill contract: the executors mira can assign work to, each declaring what
context it *consumes* and what it *produces*. This is the layer that makes task
context derivable — pipeline stages carry no I/O contract of their own
(verified: every stage exposes only run/default_settings/render_ui), so we
declare it here, seeding each stage's `produces` from summary keys actually
observed in project history.

Context assignment and executor assignment are the same graph match:
  task.needs (or task.stage)  ->  a Skill whose `produces` covers it
  that Skill's `consumes`     ->  the task's context spec
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.pipeline import STAGE_ORDER

# `produces` summary keys observed across existing projects (experiment E2).
# acquire/eda were no-ops in history -> empty, declared by behavior.
_OBSERVED_SUMMARY_KEYS: dict[str, list[str]] = {
    "business": ["notes_length"],
    "acquire": [],
    "profile": ["rows", "cols", "report_path", "minimal"],
    "eda": [],
    "clean": ["rows_before", "rows_after", "cells_imputed", "dropped_columns",
              "duplicates_dropped", "rows_dropped_outliers"],
    "feature_eng": ["new_columns", "dropped_columns", "dropped_column_reasons",
                    "encoders_saved", "restaurant_features"],
    "feature_select": ["selected_features", "selected_columns", "dropped",
                       "dropped_non_numeric", "original_features", "coerced_numeric"],
    "data_prep": ["n_train", "n_test", "scaler", "split_strategy", "test_size",
                  "stratify_used", "note"],
    "model": ["best_model", "backend", "n_models", "target", "task",
              "trained_on", "fallback_reason"],
    "evaluate": ["metrics_keys", "n_plots", "target", "task", "backend"],
    "explain": ["method", "plots", "n_samples", "shap_error"],
    "deploy": ["pipeline_path", "inference_path", "requirements_path",
               "enabled_libraries"],
}


@dataclass
class Skill:
    """An executor with a declared context contract."""

    name: str
    kind: str                       # "stage" | "ci" | "agent" | "git"
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    stage: str | None = None        # set for kind == "stage"
    summary_keys: list[str] = field(default_factory=list)


# Stages whose work is driven by dataset characteristics (preprocessing +
# analysis) — they consume the dataset-analysis facts.
_DATA_AWARE_STAGES = {
    "profile", "eda", "clean", "feature_eng", "feature_select", "data_prep", "model",
}


def _build_stage_skills() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for i, stage in enumerate(STAGE_ORDER):
        consumes = ["facts.schema", "facts.target"]
        if stage in _DATA_AWARE_STAGES:
            consumes.append("data.analysis")
        if i == 0:
            consumes.append("data.working")
        else:
            prev = STAGE_ORDER[i - 1]
            consumes += [f"{prev}.output", f"{prev}.summary"]
        keys = _OBSERVED_SUMMARY_KEYS.get(stage, [])
        produces = [f"{stage}.output", f"{stage}.summary"] + [f"{stage}.{k}" for k in keys]
        skills[stage] = Skill(
            name=f"run_stage:{stage}", kind="stage", stage=stage,
            consumes=consumes, produces=produces, summary_keys=keys,
        )
    return skills


# Registry: 12 stage skills + cross-cutting executors.
SKILLS: dict[str, Skill] = _build_stage_skills()
SKILLS["git_sync"] = Skill(
    name="git_sync", kind="git",
    consumes=[], produces=["repo.status", "repo.ci"],
)
SKILLS["gated_ci"] = Skill(
    name="gated_ci", kind="ci",
    consumes=["data_prep.output", "model.summary"],
    produces=["model.artifact", "evaluate.metrics"],
)
SKILLS["llm_agent"] = Skill(
    name="llm_agent", kind="agent",
    consumes=["intent.story", "facts.schema", "facts.target", "data.analysis",
              "repo.status"],
    produces=["code", "artifact.any"],
)


def stage_skill(stage: str) -> Skill | None:
    sk = SKILLS.get(stage)
    return sk if sk and sk.kind == "stage" else None


def match_skill(*, stage: str | None, kind: str, labels: list[str] | None = None) -> Skill:
    """Assign an executor to a task.

    A stage-linked or derived stage card binds to its stage runner; everything
    else (free-form work items) binds to the LLM/agent executor with project
    context. This never returns None — there is always a fallback executor.
    """
    if stage:
        sk = stage_skill(stage)
        if sk:
            return sk
    if kind == "stage":
        # derived stage card without an explicit stage shouldn't happen, but be safe
        return SKILLS["llm_agent"]
    return SKILLS["llm_agent"]
