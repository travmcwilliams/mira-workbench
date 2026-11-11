"""
core/investigate.py — intent-driven, schema-from-the-web project creation.

A user who has no data and only a fuzzy goal ("investigate disc-golf course
difficulty", "experiment with county grant data") should still be able to start.
This module turns that intent into a concrete gather-and-build run:

    goal + mode  ──plan──▶  topic + query spread + fields + task guess
                 ──gather─▶  discover_many (up to ~N candidates) → scrape the good ones
                 ──build──▶  consolidate scraped tables → one schema → a Project

Planning uses the ``core/llm.py`` seam (Ollama / Anthropic) but **always degrades
to a deterministic offline planner**, so it works with no model installed and
tests stay reproducible. The orchestrator takes injectable ``discover_fn`` /
``scrape_fn`` so it can be driven offline in tests without hitting the web.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# Intent modes the UI offers. Each biases the task guess and query spread.
MODES = ("investigate", "experiment", "predict", "explore")

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "by",
    "how", "what", "why", "which", "that", "this", "some", "any", "data", "dataset",
    "about", "into", "over", "from", "is", "are", "do", "i", "we", "my", "me",
    "want", "would", "like", "using", "use", "based", "most", "more", "get",
}


@dataclass
class InvestigationPlan:
    goal: str
    mode: str
    topic: str
    queries: list[str]
    fields: list[str]
    task_type: str
    target_hint: str | None = None
    provider: str = "offline"          # which planner produced this

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def _keywords(text: str, limit: int = 8) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", (text or "").lower())
    out: list[str] = []
    for t in toks:
        if t in _STOPWORDS or len(t) < 3:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _topic_from_goal(goal: str) -> str:
    kw = _keywords(goal, limit=6)
    return " ".join(kw) if kw else (goal or "").strip()[:80]


def build_query_spread(topic: str, fields: list[str], mode: str) -> list[str]:
    """A wider spread than the stock 4 queries, to reach ~100 candidates.

    Superset of ``discovery.engine.build_queries`` plus data-portal-biased and
    mode-biased variants. De-duped, order-preserving.
    """
    topic = (topic or "").strip()
    fstr = " ".join(fields[:4]) if fields else ""
    queries = [
        f"{topic} dataset csv download",
        f"{topic} data download {fstr}".strip(),
        f"{topic} statistics open data",
        f"{topic} {fstr} csv OR xlsx".strip(),
        f"{topic} data table",
        f"{topic} open data portal",
        f"{topic} .gov data",
        f"{topic} kaggle dataset",
        f"{topic} report xlsx",
        f"{topic} api data",
    ]
    if mode == "predict":
        queries.append(f"{topic} historical data csv")
    elif mode == "experiment":
        queries.append(f"{topic} sample dataset")
    # one query per field to broaden coverage
    for fld in fields[:4]:
        queries.append(f"{topic} {fld} data csv")

    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def _task_for_mode(mode: str) -> str:
    return "regression" if mode in ("predict", "experiment") else "regression"


def offline_plan(goal: str, mode: str) -> InvestigationPlan:
    """Deterministic planner — identical goal+mode → identical plan, no network."""
    mode = mode if mode in MODES else "explore"
    topic = _topic_from_goal(goal)
    fields = _keywords(goal, limit=6)
    return InvestigationPlan(
        goal=goal, mode=mode, topic=topic,
        queries=build_query_spread(topic, fields, mode),
        fields=fields, task_type=_task_for_mode(mode),
        target_hint=None, provider="offline",
    )


_PLAN_PROMPT = (
    "You are a data-sourcing planner. A user wants to start a data-science project "
    "but may not have data yet. Turn their goal into a plan for finding tabular data "
    "on the open web. Reply with STRICT JSON only, no prose:\n"
    '{"topic": str, "queries": [str, ...],  "fields": [str, ...], '
    '"task_type": "regression"|"classification", "target_hint": str}\n'
    "- topic: 2-6 word search-friendly subject.\n"
    "- queries: 8-14 diverse web-search queries likely to surface CSV/XLSX/table "
    "data (bias toward open-data portals, .gov, statistics, kaggle).\n"
    "- fields: the columns the user probably wants.\n"
    "- target_hint: the column most worth predicting/measuring, or \"\".\n\n"
    'MODE: "{mode}"\nGOAL: "{goal}"\n'
)


def plan_investigation(
    goal: str,
    mode: str = "explore",
    *,
    provider: str = "offline",
    model: str | None = None,
) -> InvestigationPlan:
    """LLM plan with a hard fall back to the deterministic offline planner."""
    mode = mode if mode in MODES else "explore"
    if provider == "offline":
        return offline_plan(goal, mode)
    try:
        from core.llm import generate

        raw = generate(_PLAN_PROMPT.format(mode=mode, goal=goal),
                       provider=provider, model=model)
        data = _parse_plan_json(raw)
        topic = str(data.get("topic") or _topic_from_goal(goal)).strip()
        fields = [str(f).strip() for f in data.get("fields", []) if str(f).strip()]
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
        if not queries:
            queries = build_query_spread(topic, fields, mode)
        task_type = data.get("task_type")
        if task_type not in ("regression", "classification"):
            task_type = _task_for_mode(mode)
        target = str(data.get("target_hint") or "").strip() or None
        return InvestigationPlan(
            goal=goal, mode=mode, topic=topic, queries=queries, fields=fields,
            task_type=task_type, target_hint=target, provider=provider,
        )
    except Exception:
        return offline_plan(goal, mode)


def _parse_plan_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("plan JSON is not an object")
    return obj


# --------------------------------------------------------------------------- #
# orchestration: plan → gather → consolidate → project
# --------------------------------------------------------------------------- #

ProgressCb = Callable[[str, float], None]  # (message, fraction 0..1)


def _default_scrape(candidate: dict[str, Any], fields: list[str]) -> Any:
    from core.acquire_bridge import ensure_acquire_env
    ensure_acquire_env()
    from discovery.probe import scrape_candidate_to_df
    return scrape_candidate_to_df(candidate, fields)


def _scrapeable(cand: dict[str, Any]) -> bool:
    """Worth a scrape attempt: probed reachable with a data url, or unprobed."""
    if cand.get("data_url"):
        return True
    if "reachable" not in cand:  # never probed — still worth trying
        return True
    return False


@dataclass
class InvestigationResult:
    plan: InvestigationPlan
    provider: str
    n_candidates: int
    n_scraped: int
    n_with_rows: int
    n_raw_search: int
    per_source: list[dict[str, Any]] = field(default_factory=list)
    report: Any = None                 # ConsolidationReport | None
    project_id: str | None = None
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        rep = self.report
        return {
            "goal": self.plan.goal,
            "mode": self.plan.mode,
            "topic": self.plan.topic,
            "planner": self.plan.provider,
            "search_provider": self.provider,
            "queries": len(self.plan.queries),
            "raw_search_urls": self.n_raw_search,
            "candidates": self.n_candidates,
            "scraped": self.n_scraped,
            "with_rows": self.n_with_rows,
            "project_id": self.project_id,
            "rows": self.rows,
            "n_columns": len(self.columns),
            "target_column": self.target_column,
            "task_type": getattr(rep, "task_type", self.plan.task_type),
            "error": self.error,
        }


@dataclass
class GatherResult:
    """The raw output of the gather phase — scraped frames + honest counts.

    Kept separate from consolidation so an interactive caller (the wizard) can
    scrape once, then re-consolidate with different cluster/column choices
    without re-hitting the web.
    """
    plan: InvestigationPlan
    provider: str
    n_candidates: int
    n_scraped: int
    n_with_rows: int
    n_raw_search: int
    per_source: list[dict[str, Any]] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)


def gather_frames(
    plan: InvestigationPlan,
    *,
    search_prefer: str | None = None,
    target_n: int = 100,
    max_scrape: int = 60,
    time_budget_s: float | None = 180.0,
    discover_fn: Callable[..., dict[str, Any]] | None = None,
    scrape_fn: Callable[[dict[str, Any], list[str]], Any] | None = None,
    progress_cb: ProgressCb | None = None,
) -> GatherResult:
    """Discover candidates for a plan and scrape the good ones into frames.

    ``discover_fn`` / ``scrape_fn`` are injectable so tests run fully offline.
    """
    from core import webschema

    def _tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, max(0.0, min(1.0, frac)))

    discover = discover_fn or webschema.discover_many
    disc = discover(plan.topic, plan.queries, plan.fields,
                    prefer=search_prefer, target_n=target_n)
    candidates: list[dict[str, Any]] = disc.get("candidates", [])
    provider = disc.get("provider", "unknown")
    _tick(f"Found {len(candidates)} candidates via {provider}", 0.25)

    g = GatherResult(plan=plan, provider=provider, n_candidates=len(candidates),
                     n_scraped=0, n_with_rows=0, n_raw_search=int(disc.get("n_raw", 0)))

    scrape = scrape_fn or _default_scrape
    to_scrape = [c for c in candidates if _scrapeable(c)][:max_scrape]
    start = time.time()
    for i, cand in enumerate(to_scrape):
        if time_budget_s is not None and (time.time() - start) > time_budget_s:
            _tick("Time budget reached — stopping scrape early", 0.6)
            break
        label = cand.get("title") or cand.get("url", "source")
        _tick(f"Scraping {i + 1}/{len(to_scrape)}: {str(label)[:60]}",
              0.25 + 0.5 * (i + 1) / max(len(to_scrape), 1))
        g.n_scraped += 1
        try:
            df = scrape(cand, plan.fields)
        except Exception as exc:  # noqa: BLE001
            g.per_source.append({"url": cand.get("url"), "rows": 0,
                                 "error": str(exc)[:160]})
            continue
        if df is None or getattr(df, "empty", True):
            g.per_source.append({"url": cand.get("url"), "rows": 0,
                                 "error": "no rows parsed"})
            continue
        df = df.copy()
        df["__source"] = str(label)[:120]
        g.frames.append(df)
        g.n_with_rows += 1
        g.per_source.append({"url": cand.get("url"), "rows": int(len(df)),
                             "cols": int(df.shape[1])})

    _tick(f"Scraped {g.n_scraped}, {g.n_with_rows} produced tables", 0.8)
    return g


def run_investigation(
    goal: str,
    mode: str = "explore",
    *,
    planner_provider: str = "offline",
    planner_model: str | None = None,
    search_prefer: str | None = None,
    target_n: int = 100,
    max_scrape: int = 60,
    time_budget_s: float | None = 180.0,
    create_project: bool = True,
    project_name: str | None = None,
    consolidate_strategy: str = "coherent",
    discover_fn: Callable[..., dict[str, Any]] | None = None,
    scrape_fn: Callable[[dict[str, Any], list[str]], Any] | None = None,
    progress_cb: ProgressCb | None = None,
) -> InvestigationResult:
    """End-to-end: plan → discover → scrape → consolidate → project.

    Honest counts are always reported, even when nothing scrapeable is found.
    """
    from core import webschema

    def _tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, max(0.0, min(1.0, frac)))

    plan = plan_investigation(goal, mode, provider=planner_provider, model=planner_model)
    _tick(f"Planned {len(plan.queries)} queries for '{plan.topic}'", 0.05)

    g = gather_frames(plan, search_prefer=search_prefer, target_n=target_n,
                      max_scrape=max_scrape, time_budget_s=time_budget_s,
                      discover_fn=discover_fn, scrape_fn=scrape_fn,
                      progress_cb=progress_cb)

    result = InvestigationResult(
        plan=plan, provider=g.provider, n_candidates=g.n_candidates,
        n_scraped=g.n_scraped, n_with_rows=g.n_with_rows,
        n_raw_search=g.n_raw_search, per_source=g.per_source,
    )

    combined, report = webschema.consolidate(
        g.frames, task_type=plan.task_type, strategy=consolidate_strategy)
    result.report = report
    result.rows = report.n_rows
    result.columns = list(combined.columns)
    result.target_column = report.target_column

    if report.n_rows == 0:
        result.error = "No candidate produced any tabular rows."
        _tick("No tabular data gathered", 1.0)
        return result

    if create_project:
        pid = build_project(plan, combined, report, project_name)
        result.project_id = pid
        _tick(f"Created project {pid} · {report.n_rows} rows × "
              f"{len(result.columns)} cols", 1.0)
    else:
        _tick("Consolidated (project not created)", 1.0)
    return result


def build_project(plan: InvestigationPlan, combined: Any, report: Any,
                  project_name: str | None = None) -> str:
    """Persist a consolidated frame + schema as a new Project. Returns its id."""
    from core.acquire_bridge import _sanitize_for_parquet
    from core.ingest import ingest_dataframe
    from core.project import Project

    combined = _sanitize_for_parquet(combined)
    name = project_name or f"{plan.mode.title()}: {plan.topic}"[:60]
    project = Project.create(name=name, task_type=report.task_type)
    ingest_dataframe(project, combined, source_name=f"web:{plan.topic}",
                     auto_profile=False)
    project.meta["schema"] = report.schema
    project.meta["task_type"] = report.task_type
    if report.target_column:
        project.meta["target_column"] = report.target_column
    project.meta["investigation"] = {
        "goal": plan.goal, "mode": plan.mode, "topic": plan.topic,
        "planner": plan.provider, "queries": plan.queries, "fields": plan.fields,
        "n_sources": report.n_sources, "target_reason": report.target_reason,
        "columns": [{"name": c.name, "dtype": c.dtype, "sources": c.sources,
                     "aliases": c.aliases} for c in report.columns],
    }
    project.save_meta()
    return str(project.id)
