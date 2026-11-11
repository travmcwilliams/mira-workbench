"""
ui/walkthrough_page.py — MIRA "Walkthrough" guided wizard.

A whiteboard of MIRA's pipeline (business → … → deploy) that you walk one step at
a time. You configure the current step's *real* options; confirming unlocks the
next step; future steps stay locked until you reach them. At the end every
selection is compiled into **valid MIRA instructions** — a task/target config plus
per-stage settings overrides — which you can save onto the project, download as
JSON, or run.

Why a wizard driven by buttons and Streamlit widgets rather than clickable graph
nodes: a graphviz ``href`` click triggers a full page reload, which wipes
``st.session_state.project`` (set on Home). Widgets + buttons rerun over the
websocket and preserve it. The graph is the *map*; the widgets are the controls.

The compiled settings use only keys each stage's ``default_settings``/``run``
actually consumes, and are persisted with ``core.stage_settings.save_stage_settings``
(which deep-merges the partial override over the stage defaults), so a subsequent
``mira_cli.py run`` honours them.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from core.library_catalog import get_enabled_libraries, get_task_config, set_task_config
from core.pipeline import STAGE_ORDER
from core.stage_settings import save_stage_settings
from ui.theme import page_hero

# --- Palette ------------------------------------------------------------------
C_START = "#2f6fbf"
C_DONE = "#5b9bd5"
C_CURRENT = "#cfe0f5"
C_LOCKED = "#e9edf2"
C_SELECTED = "#e8a33d"
C_OPTION = "#eaf1fb"
C_CHOICE = "#ffffff"
C_PICKED = "#fff1d6"
C_EDGE = "#9aa7b5"

STAGE_INFO: dict[str, tuple[str, str]] = {
    "business": ("Start", "Frame the question: what do you want to predict from this data?"),
    "acquire": ("Acquire", "Pull in / load the dataset (records, upload, or sample)."),
    "profile": ("Profile", "Auto-profile every column: types, missingness, distributions."),
    "eda": ("Explore (EDA)", "Plot relationships and spot patterns before modeling."),
    "clean": ("Clean", "Handle missing values, outliers, and duplicate rows."),
    "feature_eng": ("Feature engineering", "Encode categories; build date features."),
    "feature_select": ("Feature selection", "Keep the features that carry signal."),
    "data_prep": ("Prepare", "Split train/test and scale so evaluation is honest."),
    "model": ("Model (AutoML)", "Compare candidate models; keep the best by CV score."),
    "evaluate": ("Evaluate", "Score the winner: R²/MAE (reg) or accuracy/F1 (clf)."),
    "explain": ("Explain", "SHAP: which features drove the predictions."),
    "deploy": ("Export", "Save the trained model + pipeline for reuse."),
}


# --- Truthful candidate models (mirrors core/sklearn_automl._candidate_models) -
def candidate_model_labels(task: str, enabled: list[str]) -> list[str]:
    if task == "classification":
        out = ["Random Forest Classifier", "Extra Trees Classifier",
               "Gradient Boosting Classifier", "Logistic Regression"]
        optional = [("xgboost", "XGBoost Classifier"), ("lightgbm", "LightGBM Classifier"),
                    ("catboost", "CatBoost Classifier")]
    else:
        out = ["Random Forest Regressor", "Extra Trees Regressor",
               "Gradient Boosting Regressor", "Ridge Regression"]
        optional = [("xgboost", "XGBoost Regressor"), ("lightgbm", "LightGBM Regressor"),
                    ("catboost", "CatBoost Regressor")]
    for lib, label in optional:
        if lib in enabled:
            out.append(label)
    return out


# --- Per-step options. Each opt: key, label, kind, plus kind-specific fields. ---
# kinds: task | target | column_optional | choice | multichoice | info | info_models
# 'setting' is the real stage-settings key it compiles to (None / "__*" = special).
WIZARD_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "business": [
        {"key": "task_type", "label": "What do you want to do?", "kind": "task"},
        {"key": "target_column", "label": "Target to predict", "kind": "target"},
    ],
    "acquire": [
        {"key": "use_existing_records", "label": "Data source", "kind": "choice",
         "choices": ["use loaded data", "re-scrape source"], "default": "use loaded data",
         "setting": "use_existing_records",
         "value_map": {"use loaded data": True, "re-scrape source": False}},
    ],
    "profile": [
        {"key": "report", "label": "Profiling report", "kind": "choice",
         "choices": ["standard", "minimal (faster)"], "default": "standard",
         "setting": "minimal_threshold",
         "value_map": {"standard": 100, "minimal (faster)": 1}},
    ],
    "eda": [
        {"key": "pairplot", "label": "Pair plots", "kind": "choice",
         "choices": ["on", "off"], "default": "on", "setting": "include_pairplot",
         "value_map": {"on": True, "off": False}},
        {"key": "nplots", "label": "Plots per type", "kind": "choice",
         "choices": ["3", "5"], "default": "3", "setting": "max_numeric_plots",
         "value_map": {"3": 3, "5": 5}},
    ],
    "clean": [
        {"key": "missing", "label": "Missing values", "kind": "percolumn_missing",
         "choices": ["auto (by type)", "constant fill", "drop rows", "leave as-is"],
         "default": "auto (by type)", "grid": True},
        {"key": "outlier_method", "label": "Outlier method", "kind": "choice",
         "choices": ["iqr", "zscore", "none"], "default": "iqr", "setting": "outlier_method"},
        {"key": "outlier_action", "label": "Outlier action", "kind": "choice",
         "choices": ["clip", "drop", "flag"], "default": "clip", "setting": "outlier_action"},
        {"key": "dups", "label": "Duplicate rows", "kind": "choice",
         "choices": ["drop", "keep"], "default": "drop", "setting": "drop_duplicates",
         "value_map": {"drop": True, "keep": False}},
    ],
    "feature_eng": [
        {"key": "encoding", "label": "Categorical encoding", "kind": "percolumn_encoding",
         "choices": ["auto (by cardinality)", "one-hot all", "label all"],
         "default": "auto (by cardinality)", "grid": True},
        {"key": "dates", "label": "Date features", "kind": "choice",
         "choices": ["auto-expand", "skip"], "default": "auto-expand", "setting": "__date_policy"},
        {"key": "featuretools", "label": "Featuretools", "kind": "choice",
         "choices": ["off", "on"], "default": "off", "setting": "featuretools",
         "value_map": {"off": False, "on": True}},
    ],
    "feature_select": [
        {"key": "methods", "label": "Selection methods", "kind": "multichoice",
         "choices": ["correlation", "variance", "rfe", "importance"],
         "default": ["correlation", "variance"], "setting": "methods"},
        {"key": "corr", "label": "Correlation threshold", "kind": "choice",
         "choices": ["0.8", "0.9", "0.95"], "default": "0.9", "setting": "correlation_threshold",
         "value_map": {"0.8": 0.8, "0.9": 0.9, "0.95": 0.95}},
    ],
    "data_prep": [
        {"key": "split", "label": "Train/test split", "kind": "choice",
         "choices": ["temporal", "random"], "default": "temporal", "setting": "split_strategy"},
        {"key": "test_size", "label": "Test size", "kind": "choice",
         "choices": ["0.1", "0.2", "0.3"], "default": "0.2", "setting": "test_size",
         "value_map": {"0.1": 0.1, "0.2": 0.2, "0.3": 0.3}},
        {"key": "scaler", "label": "Feature scaling", "kind": "choice",
         "choices": ["standard", "minmax", "none"], "default": "standard", "setting": "scaler"},
        {"key": "stratify", "label": "Stratify (classification)", "kind": "choice",
         "choices": ["on", "off"], "default": "on", "setting": "stratify",
         "value_map": {"on": True, "off": False}},
        {"key": "time_column", "label": "Time column (temporal split)",
         "kind": "column_optional", "setting": "time_column"},
    ],
    "model": [
        {"key": "engine", "label": "AutoML engine", "kind": "choice",
         "choices": ["PyCaret", "sklearn"], "default": "PyCaret", "setting": "prefer_pycaret",
         "value_map": {"PyCaret": True, "sklearn": False}},
        {"key": "n_models", "label": "Models to keep", "kind": "choice",
         "choices": ["3", "5", "10"], "default": "5", "setting": "n_models",
         "value_map": {"3": 3, "5": 5, "10": 10}},
        {"key": "cv_folds", "label": "CV folds", "kind": "choice",
         "choices": ["3", "5", "10"], "default": "5", "setting": "cv_folds",
         "value_map": {"3": 3, "5": 5, "10": 10}},
        {"key": "candidates", "label": "Candidate models", "kind": "info_models"},
    ],
    "evaluate": [
        {"key": "note", "label": "Automatic", "kind": "info"},
    ],
    "explain": [
        {"key": "sample_size", "label": "SHAP sample size", "kind": "choice",
         "choices": ["200", "500", "1000"], "default": "500", "setting": "sample_size",
         "value_map": {"200": 200, "500": 500, "1000": 1000}},
    ],
    "deploy": [
        {"key": "note", "label": "Automatic", "kind": "info"},
    ],
}

_NONE_COL = "(auto / none)"

# Per-column override grids (Clean missing values, Feature-eng encoding).
_USE_POLICY = "(use policy)"
_MISSING_GRID = [_USE_POLICY, "median", "mean", "mode", "constant", "drop rows", "leave as-is"]
_ENCODING_GRID = [_USE_POLICY, "one-hot", "label"]
# UI label → the real per-column value each stage's run() consumes.
_MISSING_REAL = {"median": "median", "mean": "mean", "mode": "mode", "constant": "constant",
                 "drop rows": "drop_rows", "leave as-is": "none"}
_ENCODING_REAL = {"one-hot": "onehot", "label": "label"}
# Inverse maps, for resuming a saved plan back into the grid.
_MISSING_REAL_INV = {v: k for k, v in _MISSING_REAL.items()}
_ENCODING_REAL_INV = {v: k for k, v in _ENCODING_REAL.items()}


# --- helpers ------------------------------------------------------------------
def _esc(text: Any) -> str:
    s = str(text)
    return s.replace("\\", "\\\\").replace('"', "'").replace("\n", " ").replace("\r", " ").strip()


def _dataset_columns(project: Any) -> list[str]:
    schema = project.meta.get("schema") or {}
    if isinstance(schema, dict) and schema:
        return list(schema.keys())
    if not project.has_raw_data():
        return []
    path = project.root / "data" / "working.parquet"
    if not path.exists():
        path = project.root / "data" / "raw.parquet"
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return list(project.get_working_data().columns)
        except Exception:
            return []


def _active_task(project: Any, confirmed: dict[str, dict]) -> str:
    return confirmed.get("business", {}).get("task_type") or project.meta.get("task_type", "regression")


def _summary(stage: str, confirmed: dict[str, dict]) -> str:
    ch = confirmed.get(stage, {})
    parts: list[str] = []
    for opt in WIZARD_OPTIONS.get(stage, []):
        if opt["kind"] in ("info", "info_models"):
            continue
        v = ch.get(opt["key"])
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            v = ", ".join(v)
        parts.append(str(v))
        if len(parts) >= 2:
            break
    return " · ".join(parts)


# --- graph --------------------------------------------------------------------
def build_wizard_dot(
    project: Any, *, idx: int, confirmed: dict[str, dict], live: dict[str, Any],
    active_task: str,
) -> str:
    meta = project.meta
    enabled = get_enabled_libraries(project)
    target = meta.get("target_column")
    columns = _dataset_columns(project)
    cur_stage = STAGE_ORDER[idx] if 0 <= idx < len(STAGE_ORDER) else None

    lines = [
        "digraph mira_wizard {",
        "  rankdir=LR;",
        '  bgcolor="white";',
        '  graph [splines=spline, nodesep=0.3, ranksep=0.8, fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
        'color="#aab4c0", penwidth=1.2];',
        f'  edge [color="{C_EDGE}", arrowsize=0.7];',
        "  subgraph cluster_legend {",
        '    label="Legend"; labeljust="l"; fontsize=11; color="#c4ccd6";',
        f'    lg_done [label="Done", fillcolor="{C_DONE}", fontcolor="white"];',
        f'    lg_cur  [label="Current step", fillcolor="{C_CURRENT}", color="{C_SELECTED}", penwidth=2.6];',
        f'    lg_lock [label="Locked", fillcolor="{C_LOCKED}", fontcolor="#9aa7b5"];',
        f'    lg_pick [label="Your choice", fillcolor="{C_PICKED}", color="{C_SELECTED}", penwidth=2.2];',
        "    lg_done -> lg_cur -> lg_lock -> lg_pick [style=invis];",
        "  }",
    ]

    for i, stage in enumerate(STAGE_ORDER):
        title, desc = STAGE_INFO[stage]
        is_cur = i == idx
        is_done = (stage in confirmed) and not is_cur
        if stage == "business":
            tgt = f" → {_esc(target)}" if target else ""
            label = f"What do you want\\nto do?{tgt}"
            fill, font = C_START, "white"
        elif is_done:
            summ = _summary(stage, confirmed)
            label = f"{_esc(title)}\\n✓ {_esc(summ)}" if summ else f"{_esc(title)}\\n✓"
            fill, font = C_DONE, "white"
        elif is_cur:
            label = _esc(title)
            fill, font = C_CURRENT, "black"
        else:
            label = f"🔒 {_esc(title)}"
            fill, font = C_LOCKED, "#9aa7b5"
        border = f', color="{C_SELECTED}", penwidth=3.0' if is_cur else ""
        lines.append(
            f'  s_{stage} [label="{label}", fillcolor="{fill}", fontcolor="{font}", '
            f'tooltip="{_esc(desc)}"{border}];'
        )
    lines.append("  " + " -> ".join(f"s_{s}" for s in STAGE_ORDER) + ";")

    # fan out the current step's options, highlighting the live choice
    if cur_stage is not None:
        for ci, opt in enumerate(WIZARD_OPTIONS.get(cur_stage, [])):
            kind = opt["kind"]
            if kind == "info":
                continue
            cat_id = f"opt_{cur_stage}_{ci}"
            lines.append(
                f'  {cat_id} [label="{_esc(opt["label"])}", fillcolor="{C_OPTION}", color="#9bb6dd"];'
            )
            lines.append(f"  s_{cur_stage} -> {cat_id};")

            if kind == "info_models":
                for cj, ch in enumerate(candidate_model_labels(active_task, enabled)):
                    lines.append(f'  {cat_id}_{cj} [label="{_esc(ch)}", fillcolor="{C_OPTION}"];')
                    lines.append(f"  {cat_id} -> {cat_id}_{cj};")
                continue

            if kind in ("target", "column_optional"):
                val = live.get(opt["key"])
                choices, picked = [str(val) if val not in (None, "") else "—"], {str(val)}
            elif kind == "multichoice":
                choices, picked = opt["choices"], set(live.get(opt["key"]) or [])
            else:  # choice, task, percolumn_*
                choices = opt.get("choices", [])
                picked = {live.get(opt["key"])}

            for cj, ch in enumerate(choices):
                sel = ch in picked
                fill = C_PICKED if sel else C_CHOICE
                bd = f', color="{C_SELECTED}", penwidth=2.2' if sel else ""
                tick = "  ✓" if sel else ""
                lines.append(f'  {cat_id}_{cj} [label="{_esc(ch)}{tick}", fillcolor="{fill}"{bd}];')
                lines.append(f"  {cat_id} -> {cat_id}_{cj};")

    lines.append("}")
    return "\n".join(lines)


# --- compile selections into MIRA instructions --------------------------------
def compile_instructions(
    project: Any, confirmed: dict[str, dict]
) -> tuple[dict[str, Any], dict[str, dict]]:
    """Return (top_level {task_type,target_column}, {stage: settings_override})."""
    schema = project.meta.get("schema") or {}
    all_cols = list(schema.keys()) or _dataset_columns(project)
    # The target is never an input feature, so it must not be encoded (mirrors
    # stages/feature_eng.default_settings, which skips col == target).
    target_col = confirmed.get("business", {}).get("target_column") or project.meta.get("target_column")
    cat_cols = [c for c, d in schema.items() if d == "categorical" and c != target_col] if schema else []
    top: dict[str, Any] = {}
    stage_settings: dict[str, dict] = {}

    for stage in STAGE_ORDER:
        ch = confirmed.get(stage, {})
        if not ch:
            continue
        sset: dict[str, Any] = {}
        for opt in WIZARD_OPTIONS.get(stage, []):
            kind, key = opt["kind"], opt["key"]
            if key not in ch:
                continue
            val = ch[key]
            if kind == "task":
                top["task_type"] = val
            elif kind == "target":
                if val and not str(val).startswith("("):
                    top["target_column"] = val
            elif kind == "multichoice":
                sset[opt["setting"]] = list(val)
            elif kind == "column_optional":
                sset[opt["setting"]] = None if val in (None, _NONE_COL) else val
            elif kind == "percolumn_missing":
                mapping = {"constant fill": "constant", "drop rows": "drop_rows",
                           "leave as-is": "none"}
                base = {c: mapping[val] for c in all_cols} if val in mapping and all_cols else {}
                # Per-column overrides win over the global policy.
                for col, ui_strat in (ch.get(key + "_overrides") or {}).items():
                    real = _MISSING_REAL.get(ui_strat)
                    if real:
                        base[col] = real
                if base:
                    sset["missing_strategy"] = base
            elif kind == "percolumn_encoding":
                emap = {"one-hot all": "onehot", "label all": "label"}
                base = {c: emap[val] for c in cat_cols} if val in emap and cat_cols else {}
                for col, ui_meth in (ch.get(key + "_overrides") or {}).items():
                    if col == target_col:
                        continue  # never encode the target
                    real = _ENCODING_REAL.get(ui_meth)
                    if real:
                        base[col] = real
                if base:
                    sset["encoding"] = base
            elif kind == "choice":
                setting = opt.get("setting")
                v = opt.get("value_map", {}).get(val, val)
                if setting == "__date_policy":
                    if val == "skip":
                        sset["date_features"] = []
                elif setting:
                    sset[setting] = v
        if sset:
            stage_settings[stage] = sset
    return top, stage_settings


def save_instructions(project: Any, top: dict[str, Any], stage_settings: dict[str, dict]) -> None:
    if "task_type" in top or "target_column" in top:
        set_task_config(
            project,
            top.get("task_type", project.meta.get("task_type", "regression")),
            top.get("target_column", project.meta.get("target_column")),
        )
    for stage, sset in stage_settings.items():
        save_stage_settings(project, stage, sset)


# --- widgets ------------------------------------------------------------------
def _init_defaults(stage: str, project: Any, columns: list[str]) -> None:
    detected = project.meta.get("task_type", "regression")
    cur_target = project.meta.get("target_column")
    for opt in WIZARD_OPTIONS.get(stage, []):
        k = f"wt_w_{stage}_{opt['key']}"
        if k in st.session_state:
            continue
        kind = opt["kind"]
        if kind == "task":
            st.session_state[k] = detected
        elif kind == "target":
            st.session_state[k] = cur_target if cur_target in columns else (columns[0] if columns else "(load data to set target)")
        elif kind == "column_optional":
            st.session_state[k] = _NONE_COL
        elif kind == "multichoice":
            st.session_state[k] = list(opt.get("default", []))
        elif kind in ("info", "info_models"):
            continue
        else:
            st.session_state[k] = opt.get("default")


def _render_widgets(stage: str, project: Any, columns: list[str], active_task: str,
                    enabled: list[str]) -> None:
    any_input = False
    for opt in WIZARD_OPTIONS.get(stage, []):
        kind, k = opt["kind"], f"wt_w_{stage}_{opt['key']}"
        help_ = opt.get("help")
        if kind == "task":
            st.radio(opt["label"], ["regression", "classification"], key=k, horizontal=True); any_input = True
        elif kind == "target":
            opts = columns or ["(load data to set target)"]
            st.selectbox(opt["label"], opts, key=k); any_input = True
        elif kind == "column_optional":
            st.selectbox(opt["label"], [_NONE_COL] + (columns or []), key=k,
                         help="Only used when split = temporal."); any_input = True
        elif kind == "multichoice":
            st.multiselect(opt["label"], opt["choices"], key=k); any_input = True
        elif kind == "choice":
            st.radio(opt["label"], opt["choices"], key=k, horizontal=True, help=help_); any_input = True
        elif kind == "info_models":
            st.caption(f"**Candidate models** MIRA will compare for *{active_task}*:")
            st.write(", ".join(candidate_model_labels(active_task, enabled)))
        elif kind == "info":
            st.info("No choices here — MIRA handles this step automatically.")
    if not any_input and stage in ("evaluate", "deploy"):
        pass


# --- main render --------------------------------------------------------------
def render_walkthrough_page(project: Any | None) -> None:
    page_hero(
        "🗺️ Guided Walkthrough",
        "Walk the pipeline one step at a time — configure each step, confirming "
        "unlocks the next. At the end your choices compile into MIRA instructions "
        "you can save, download, or run.",
    )

    if project is None:
        st.info("Select (or create) a project on the **Home** page first — the wizard is driven by your data, task, and target.")
        _render_demo()
        return

    columns = _dataset_columns(project)
    detected = get_task_config(project)["task_type"]
    enabled = get_enabled_libraries(project)

    st.session_state.setdefault("wt_idx", 0)
    st.session_state.setdefault("wt_choices", {})
    st.session_state.setdefault("wt_finished", False)
    confirmed: dict[str, dict] = st.session_state.wt_choices

    # top controls
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.progress(
            (len(confirmed) / len(STAGE_ORDER)) if not st.session_state.wt_finished else 1.0,
            text=f"{len(confirmed)} / {len(STAGE_ORDER)} steps configured",
        )
    with tcol2:
        if st.button("↻ Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("wt_"):
                    del st.session_state[key]
            st.rerun()

    if st.session_state.wt_finished:
        _render_finish(project, confirmed, enabled)
        return

    idx = min(st.session_state.wt_idx, len(STAGE_ORDER) - 1)
    cur_stage = STAGE_ORDER[idx]
    _init_defaults(cur_stage, project, columns)

    live = {opt["key"]: st.session_state.get(f"wt_w_{cur_stage}_{opt['key']}")
            for opt in WIZARD_OPTIONS.get(cur_stage, []) if opt["kind"] not in ("info", "info_models")}
    active_task = _active_task(project, confirmed)
    if cur_stage == "business":
        active_task = live.get("task_type", active_task)

    # the map
    dot = build_wizard_dot(project, idx=idx, confirmed=confirmed, live=live, active_task=active_task)
    st.graphviz_chart(dot, use_container_width=True)

    # step navigator (jump back to any configured step)
    _step_chips(idx, confirmed)

    # current-step panel
    title, desc = STAGE_INFO[cur_stage]
    st.markdown(f"### Step {idx + 1} of {len(STAGE_ORDER)} — {title}")
    st.write(desc)
    _render_widgets(cur_stage, project, columns, active_task, enabled)

    # nav buttons
    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        if st.button("← Back", disabled=idx == 0, use_container_width=True):
            st.session_state.wt_idx = max(0, idx - 1)
            st.rerun()
    with b2:
        is_last = idx == len(STAGE_ORDER) - 1
        label = "Finish ✓" if is_last else "Confirm & next →"
        if st.button(label, type="primary", use_container_width=True):
            _commit(cur_stage)
            if is_last:
                st.session_state.wt_finished = True
            else:
                st.session_state.wt_idx = idx + 1
            st.rerun()


def _commit(stage: str) -> None:
    vals: dict[str, Any] = {}
    for opt in WIZARD_OPTIONS.get(stage, []):
        if opt["kind"] in ("info", "info_models"):
            continue
        vals[opt["key"]] = st.session_state.get(f"wt_w_{stage}_{opt['key']}")
    st.session_state.wt_choices[stage] = vals


def _step_chips(idx: int, confirmed: dict[str, dict]) -> None:
    st.caption("Jump to a configured step:")
    half = (len(STAGE_ORDER) + 1) // 2
    for row in (STAGE_ORDER[:half], STAGE_ORDER[half:]):
        cols = st.columns(len(row))
        for col, stage in zip(cols, row):
            i = STAGE_ORDER.index(stage)
            title = STAGE_INFO[stage][0]
            if i == idx:
                mark, disabled = "▸", True
            elif stage in confirmed:
                mark, disabled = "✓", False
            else:
                mark, disabled = "🔒", True
            if col.button(f"{mark} {title}", key=f"wt_chip_{stage}", disabled=disabled,
                          use_container_width=True):
                st.session_state.wt_idx = i
                st.rerun()


def _render_finish(project: Any, confirmed: dict[str, dict], enabled: list[str]) -> None:
    st.success("All steps configured. Here are your **MIRA instructions**.")
    top, stage_settings = compile_instructions(project, confirmed)

    st.markdown("#### Plan")
    task = top.get("task_type", project.meta.get("task_type", "regression"))
    target = top.get("target_column", project.meta.get("target_column"))
    st.markdown(f"- **Task:** {task}  ·  **Target:** {target or '— not set'}")
    for stage in STAGE_ORDER:
        if stage in stage_settings:
            kvs = ", ".join(f"`{k}` = {v}" for k, v in stage_settings[stage].items())
            st.markdown(f"- **{STAGE_INFO[stage][0]}** → {kvs}")
    st.markdown(
        f"- **Model** will compare: {', '.join(candidate_model_labels(task, enabled))}"
    )

    instructions = {"task_type": task, "target_column": target, "stage_settings": stage_settings}
    with st.expander("Machine-readable instructions (JSON)", expanded=False):
        st.code(json.dumps(instructions, indent=2), language="json")

    cli = f"python mira_cli.py run --project-id {project.id} --through evaluate"
    st.markdown("**Equivalent CLI (after saving):**")
    st.code(cli, language="bash")

    a, b, c = st.columns(3)
    with a:
        if st.button("💾 Save to project", type="primary", use_container_width=True):
            save_instructions(project, top, stage_settings)
            st.success("Saved task config + per-stage settings to the project.")
    with b:
        st.download_button(
            "📥 Download JSON", json.dumps(instructions, indent=2),
            file_name=f"mira_plan_{project.id}.json", mime="application/json",
            use_container_width=True,
        )
    with c:
        run_clicked = st.button("▶ Save & run pipeline", use_container_width=True,
                                help="Saves, then runs acquire→…→evaluate. May take a while.")

    if run_clicked:
        save_instructions(project, top, stage_settings)
        import time

        from core.pipeline import Pipeline
        from ui.stage_results import render_stage_result, snapshot_stats

        pipe = Pipeline(project)
        st.divider()
        st.subheader("Live results")
        st.caption(
            "Each step's findings and the standard graphs for getting data ready "
            "for preprocessing appear below, the moment that step completes."
        )
        progress = st.status("Running pipeline…", expanded=False)
        for stage in STAGE_ORDER:
            if stage in ("business", "acquire") and not project.has_raw_data():
                continue
            progress.update(label=f"Running: {STAGE_INFO[stage][0]}…", state="running")
            before = snapshot_stats(project)
            t0 = time.time()
            res = pipe.run_stage(stage)
            dur = int((time.time() - t0) * 1000)
            with st.container(border=True):
                render_stage_result(
                    project, stage, res, before=before, duration_ms=dur,
                    key=f"wtrun_{stage}",
                )
            if res.status != "success":
                progress.update(label=f"Stopped at {STAGE_INFO[stage][0]}", state="error")
                break
            if stage == "evaluate":
                progress.update(label="Pipeline ran through evaluate ✓", state="complete")
                break

    st.divider()
    if st.button("↩ Edit selections"):
        st.session_state.wt_finished = False
        st.session_state.wt_idx = len(STAGE_ORDER) - 1
        st.rerun()


def _render_demo() -> None:
    dot = "\n".join([
        "digraph demo {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#aab4c0"];',
        f'  start [label="What do you want\\nto do?", fillcolor="{C_START}", fontcolor="white"];',
        f'  clean [label="Clean", fillcolor="{C_CURRENT}", color="{C_SELECTED}", penwidth=3.0];',
        f'  model [label="🔒 Model", fillcolor="{C_LOCKED}", fontcolor="#9aa7b5"];',
        "  start -> clean -> model;",
        f'  o1 [label="Missing values", fillcolor="{C_OPTION}"];',
        "  clean -> o1;",
        f'  c1 [label="auto ✓", fillcolor="{C_PICKED}", color="{C_SELECTED}", penwidth=2.2];',
        f'  c2 [label="drop rows", fillcolor="white"];',
        "  o1 -> c1; o1 -> c2;",
        "}",
    ])
    st.graphviz_chart(dot, use_container_width=True)
