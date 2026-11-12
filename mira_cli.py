"""MIRA headless CLI: acquire sources and run the DS pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.acquire_bridge import (
    discover_sources,
    ingest_source_id,
    pull_source_into_project,
    scrape_candidates_into_project,
)
from core.data_loader import detect_schema, load_file, suggest_target
from core.ingest import ingest_dataframe
from core.pipeline import STAGE_ORDER, Pipeline
from core.project import Project
from core.stage_settings import get_stage_settings, save_stage_settings


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII output (e.g. the board's ``▸``
    glyph) never crashes on a legacy Windows console (cp1252). ``errors``
    downgrades to a replacement char instead of raising if UTF-8 is somehow
    unavailable, so printing can never abort the command.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _ingest_no_profile(project, df, **kwargs):
    return ingest_dataframe(project, df, auto_profile=False, **kwargs)


def _load_data(args: argparse.Namespace, project: Project) -> tuple[object, int | None]:
    """Load data into project. Returns (df, error_code or None)."""
    if args.from_records:
        if not args.source_id:
            print("ERROR: --from-records requires --source-id", file=sys.stderr)
            return None, 1
        project.meta["acquire_source_id"] = args.source_id
        project.save_meta()
        try:
            summary = pull_source_into_project(
                project,
                args.source_id,
                scrape=not args.use_existing_records,
                ingest_fn=_ingest_no_profile,
            )
        except Exception as exc:
            print(f"ERROR: records pull failed: {exc}", file=sys.stderr)
            return None, 1
        df = project.get_working_data()
        print(f"From records: {summary['rows']} rows, {len(summary['columns'])} cols")

        if not args.skip_acquire_stage:
            pipe = Pipeline(project)
            acq = pipe.run_stage(
                "acquire",
                {
                    "source_id": args.source_id,
                    "use_existing_records": True,
                },
            )
            if acq.status != "success":
                print(f"WARNING: acquire stage: {acq.error}", file=sys.stderr)
            df = project.get_working_data()
        return df, None

    if not args.data:
        print("ERROR: pass --data or --from-records", file=sys.stderr)
        return None, 1
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return None, 1
    df = load_file(data_path)
    err = ingest_dataframe(project, df, source_name=data_path.name, auto_profile=False)
    if err:
        print(f"WARNING: ingest note: {err}")
    return df, None


def _run_pipeline(
    project: Project,
    *,
    from_stage: str | None,
    until: str | None,
    through: str | None,
) -> tuple[list, int]:
    stop_at = through or until
    pipeline = Pipeline(project)
    results = []

    start_idx = 0
    if from_stage:
        if from_stage not in STAGE_ORDER:
            print(f"ERROR: unknown --from-stage: {from_stage}", file=sys.stderr)
            return [], 1
        start_idx = STAGE_ORDER.index(from_stage)

    for name in STAGE_ORDER[start_idx:]:
        settings = get_stage_settings(project, name)
        result = pipeline.run_stage(name, settings)
        results.append((name, result))
        status = result.status.upper()
        summary = result.summary or {}
        extra = f" | {summary}" if summary else ""
        print(f"{name:16} {status}{extra}")
        if result.status == "error":
            err = (result.error or "unknown error").strip()
            print(f"  -> {err[:500]}")
            return results, 1
        if stop_at and name == stop_at:
            break

    return results, 0


def cmd_acquire(args: argparse.Namespace) -> int:
    try:
        result = ingest_source_id(args.source_id, dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: acquire failed: {exc}", file=sys.stderr)
        return 1

    print(f"Source: {result['source_id']}")
    print(f"Rows loaded: {result['rows_loaded']}")
    print(f"Records: {result['records_before']:,} → {result['records_after']:,}")
    if args.dry_run:
        print("(dry run — no DB writes)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.project_id:
        project = Project.load(args.project_id)
        print(f"Loaded project: {project.id} ({project.meta.get('name')})")
    else:
        project = Project.create(name=args.name, task_type=args.task)
        print(f"Created project: {project.id} ({args.name})")

    # A project that already has data (e.g. created by `discover`) and no new
    # data source specified: run on what's there instead of erroring.
    if args.project_id and project.has_raw_data() and not args.data and not args.from_records:
        df = project.get_working_data()
        print(f"Using existing project data: {len(df)} rows x {len(df.columns)} cols")
    else:
        df, err_code = _load_data(args, project)
        if err_code:
            return err_code
    assert df is not None

    schema = detect_schema(df)
    project.meta["schema"] = schema

    task_type = args.task
    target = args.target
    if args.suggest_target or not target:
        suggestion = suggest_target(df, schema, task_type=task_type)
        if not target and suggestion.get("target_column"):
            target = suggestion["target_column"]
            print(f"Suggested target: {target} ({suggestion.get('reason', 'auto')})")
        if suggestion.get("task_type"):
            task_type = suggestion["task_type"]

    if not target:
        print("ERROR: no target column. Pass --target or use --suggest-target.", file=sys.stderr)
        return 1

    project.meta["task_type"] = task_type
    project.meta["target_column"] = target
    project.save_meta()

    # Wire the train/test split strategy into pipeline core for every CLI run.
    # data_prep now defaults to a leak-safe temporal split, so a bare
    # `run` is temporal with no flag; --split-strategy / --time-column override
    # and are persisted to stage_settings (provenance) so downstream consumers and
    # repeat runs reproduce the same split. The effective strategy is always
    # surfaced so no run silently look-ahead-leaks.
    if getattr(args, "split_strategy", None) or args.time_column:
        data_prep_settings: dict[str, object] = {}
        if args.split_strategy:
            data_prep_settings["split_strategy"] = args.split_strategy
        if args.time_column:
            data_prep_settings["time_column"] = args.time_column
        save_stage_settings(project, "data_prep", data_prep_settings)

    effective = get_stage_settings(project, "data_prep")
    eff_split = str(effective.get("split_strategy", "temporal"))
    eff_time = effective.get("time_column")
    tag = "  [leak-safe default]" if not getattr(args, "split_strategy", None) and eff_split == "temporal" else ""
    print(f"Split strategy: {eff_split}"
          + (f" (ordered by '{eff_time}')" if eff_time else "")
          + tag)

    print(f"Data: {len(df)} rows x {len(df.columns)} cols")
    print(f"Task: {task_type} | Target: {target}")
    print("-" * 60)

    from_stage = args.from_stage
    if from_stage is None and not args.from_records and args.data:
        from_stage = "profile"

    results, code = _run_pipeline(
        project,
        from_stage=from_stage,
        until=args.until,
        through=args.through,
    )

    completed = project.meta.get("completed_stages", [])
    print("-" * 60)
    print(f"Completed stages: {', '.join(completed) or '(none)'}")
    print(f"Project folder: {project.root.resolve()}")
    return code if results else 1


def _parse_indices(raw: str | None, n: int) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 0 <= int(part) < n:
            out.append(int(part))
    return out


def cmd_discover(args: argparse.Namespace) -> int:
    fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()]
    try:
        result = discover_sources(
            args.topic,
            fields,
            prefer=args.prefer,
            max_candidates=args.max,
            do_probe=not args.no_probe,
        )
    except Exception as exc:
        print(f"ERROR: discovery failed: {exc}", file=sys.stderr)
        return 1

    candidates = result["candidates"]
    print(f"Provider: {result['provider']}  |  {len(candidates)} candidates")
    print("-" * 70)
    for i, c in enumerate(candidates):
        matched = c.get("matched_fields") or []
        cov = f"{len(matched)}/{len(fields)}" if fields else "-"
        print(f"[{i:2}] score={c.get('score',0):4}  {c.get('scraper_type','?'):17} "
              f"fmt={str(c.get('data_format') or '-'):10} fields={cov}")
        print(f"     {(c.get('title') or '')[:80]}")
        print(f"     {c.get('url','')}")

    idx = _parse_indices(args.scrape, len(candidates))
    if args.scrape_top:
        idx = sorted(set(idx) | set(range(min(args.scrape_top, len(candidates)))))
    if not idx:
        return 0

    chosen = [candidates[i] for i in idx]
    print("-" * 70)
    print(f"Scraping {len(chosen)} selected source(s)…")
    summary = scrape_candidates_into_project(
        chosen, topic=args.topic, fields=fields, task_type=args.task
    )
    for s in summary.get("sources", []):
        print(f"  {s.get('rows',0):>6} rows  {s.get('url','')[:60]}"
              + (f"  ({s['error']})" if s.get("error") else ""))

    pid = summary.get("project_id")
    if not pid:
        print(f"ERROR: {summary.get('error', 'no data scraped')}", file=sys.stderr)
        return 1
    print(f"Project: {pid}  |  {summary['rows']} rows × {len(summary['columns'])} cols")
    print(f"Suggested target: {summary.get('target_column') or '(none)'} "
          f"({summary.get('target_reason', 'n/a')})")

    if args.through:
        project = Project.load(pid)
        print("-" * 70)
        _, code = _run_pipeline(project, from_stage="profile", until=None, through=args.through)
        print(f"Project folder: {project.root.resolve()}")
        return code
    print(f"Run the pipeline with: python mira_cli.py run --project-id {pid} --through model")
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    """Intent-driven: goal + mode -> scrape the web -> a project with a schema."""
    from core.investigate import run_investigation

    def _progress(msg: str, frac: float) -> None:
        print(f"  [{int(frac * 100):3}%] {msg}")

    try:
        result = run_investigation(
            args.goal,
            args.mode,
            planner_provider=args.planner,
            planner_model=args.model,
            search_prefer=args.prefer,
            target_n=args.target_n,
            max_scrape=args.max_scrape,
            time_budget_s=args.time_budget,
            create_project=not args.no_create,
            progress_cb=None if args.json else _progress,
        )
    except Exception as exc:
        print(f"ERROR: investigation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json as _json
        print(_json.dumps(result.summary(), indent=2))
    else:
        s = result.summary()
        print("-" * 70)
        print(f"Goal      : {s['goal']}")
        print(f"Mode      : {s['mode']}   Topic: {s['topic']}   Planner: {s['planner']}")
        print(f"Search    : {s['search_provider']}  |  {s['queries']} queries  "
              f"-> {s['raw_search_urls']} urls  -> {s['candidates']} candidates")
        print(f"Scraped   : {s['scraped']}   with tables: {s['with_rows']}")
        if result.report is not None:
            print(f"Schema    : {s['rows']} rows × {s['n_columns']} cols  "
                  f"(task {s['task_type']})")
            top = [c for c in result.report.columns][:12]
            for c in top:
                print(f"    {c.name:24} {c.dtype:12} from {c.sources} source(s)")
        if s.get("project_id"):
            print(f"Project   : {s['project_id']}   target: {s['target_column'] or '(none)'}")
            print(f"Next      : python mira_cli.py run --project-id {s['project_id']} --through model")
        if s.get("error"):
            print(f"Note      : {s['error']}")
    return 0 if (args.no_create or result.project_id) else 1


def cmd_board(args: argparse.Namespace) -> int:
    from core import board

    try:
        project = Project.load(args.project_id)
    except FileNotFoundError:
        print(f"ERROR: project not found: {args.project_id}", file=sys.stderr)
        return 1

    if args.board_action == "add":
        labels = [s.strip() for s in (args.labels or "").split(",") if s.strip()]
        tid = board.add_task(
            project,
            args.title,
            description=args.description,
            column=args.column,
            priority=args.priority,
            points=args.points,
            assignee=args.assignee,
            labels=labels or None,
        )
        print(f"Added task #{tid}: {args.title}")
        return 0

    if args.board_action == "move":
        if board.state.get_task(project.root / "state.db", args.task) is None:
            print(f"ERROR: task #{args.task} not found", file=sys.stderr)
            return 1
        board.move_task(project, args.task, args.to)
        print(f"Moved task #{args.task} -> {args.to}")
        return 0

    if args.board_action == "run":
        from core import executor

        row = board.state.get_task(project.root / "state.db", args.task)
        if row is None:
            print(f"ERROR: task #{args.task} not found", file=sys.stderr)
            return 1
        task = board._row_to_task(row)
        res = executor.execute_task(project, task, provider=args.provider,
                                    dry_run=args.dry_run)
        print(f"Task #{args.task}: {task.title}")
        print(f"  executor: {res.skill} ({res.kind})")
        print(f"  status  : {res.status.upper()}")
        if res.summary:
            print(f"  summary : {res.summary}")
        if res.error:
            print(f"  error   : {res.error}")
        if res.output and (args.show_output or res.kind == "agent"):
            print("-" * 60)
            print(res.output)
        return 0 if res.ok else 1

    if args.board_action == "context":
        from core import context as context_mod

        row = board.state.get_task(project.root / "state.db", args.task)
        if row is None:
            print(f"ERROR: task #{args.task} not found", file=sys.stderr)
            return 1
        task = board._row_to_task(row)
        ctx = context_mod.get_context(project, task, refresh=args.refresh)
        if args.brief:
            print(context_mod.render_brief(ctx))
        else:
            stale = context_mod.is_stale(project, task)
            print(f"Task #{args.task}: {task.title}")
            print(f"  executor : {ctx['skill']} ({ctx['skill_kind']})")
            print(f"  needs    : {', '.join(ctx['needs']) or '(none)'}")
            print(f"  produces : {', '.join(ctx['produces']) or '(none)'}")
            print(f"  fingerprint: {ctx['fingerprint']}  ({'STALE' if stale else 'fresh'})")
        return 0

    # default: list
    full = board.get_board(project, include_stage_cards=not args.tasks_only)
    columns = board.board_columns(project)
    for col in columns:
        cards = full.get(col, [])
        print(f"\n[{col}]  ({len(cards)})")
        for t in cards:
            marker = "▸" if t.kind == "stage" else "#" + str(t.id)
            pts = f" {t.points}pt" if t.points else ""
            prio = "" if t.priority == "medium" else f" !{t.priority}"
            print(f"  {marker:>5}  {t.title}{pts}{prio}")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    try:
        project = Project.load(args.project_id)
    except FileNotFoundError:
        print(f"ERROR: project not found: {args.project_id}", file=sys.stderr)
        return 1

    if args.data_action == "explore":
        return _cmd_data_explore(args, project)

    from core import dataset_analysis
    analysis = dataset_analysis.analyze_project(project, force=args.force)
    if not analysis.get("available"):
        print(f"No dataset analysis: {analysis.get('note', 'no working data')}")
        return 1

    if args.json:
        import json as _json
        print(_json.dumps(analysis if args.full
                          else dataset_analysis.compact_summary(analysis), indent=2))
    else:
        print(dataset_analysis.render_markdown(analysis))
    return 0


def _cmd_data_explore(args: argparse.Namespace, project: Project) -> int:
    from core import explore

    if not project.has_raw_data():
        print("No working data to explore.", file=sys.stderr)
        return 1
    df = project.get_working_data()
    recipes = explore.suggest_recipes(
        df, project.meta.get("schema"), project.meta.get("target_column"))

    print(f"Suggested analyses ({len(recipes)}):")
    for r in recipes:
        print(f"  - {r.key:14} {r.title}  [{r.concept}]")

    if args.run:
        rec = next((r for r in recipes if r.key == args.run), None)
        if rec is None:
            print(f"ERROR: no applicable recipe '{args.run}'", file=sys.stderr)
            return 1
        res = explore.run_recipe(df, rec)
        print(f"\n[{rec.key}] {res.status}"
              + (f" — {res.text}" if res.text else "")
              + (f"  ERROR: {res.error}" if res.error else ""))

    if args.notebook:
        path = explore.export_notebook(project, recipes)
        print(f"\nNotebook written: {path}")
    return 0


def cmd_sprint(args: argparse.Namespace) -> int:
    from core import sprint as sprint_mod

    try:
        project = Project.load(args.project_id)
    except FileNotFoundError:
        print(f"ERROR: project not found: {args.project_id}", file=sys.stderr)
        return 1

    plan = sprint_mod.plan_sprint(
        project,
        capacity=args.capacity,
        provider=args.provider,
        materialize=args.materialize,
    )

    if args.export == "json":
        print(sprint_mod.export_json(plan))
    elif args.export == "md":
        print(sprint_mod.export_markdown(plan))
    else:
        print(f"Sprint plan ({plan.get('provider')}) - {len(plan.get('stories', []))} stories")
        print(f"Goal: {plan.get('goal', '')}")
        for i, s in enumerate(plan.get("stories", []), 1):
            print(f"  {i:2}. [{s.get('points')}pt/{s.get('priority')}] {s.get('title')}")
        if plan.get("sprint_id"):
            print(f"Saved as sprint #{plan['sprint_id']}"
                  + ("  (stories materialized to board)" if args.materialize else ""))
    return 0


def cmd_ci(args: argparse.Namespace) -> int:
    from core.ci import run_ci

    try:
        project = Project.load(args.project_id)
    except FileNotFoundError:
        print(f"ERROR: project not found: {args.project_id}", file=sys.stderr)
        return 1

    print(f"CI run: {project.id} ({project.meta.get('name')})")
    print("-" * 60)
    result = run_ci(project, through=args.through, from_stage=args.from_stage)

    for gr in result.gate_results:
        icon = {"pass": "✓", "fail": "✗", "skip": "·"}.get(gr.status, "?")
        print(f"  [{icon}] {gr.gate}  ({gr.message})")

    print("-" * 60)
    print(f"Stages run: {', '.join(result.stages_run) or '(none)'}")
    if result.status == "passed":
        print("CI PASSED")
        return 0
    if result.status == "failed":
        g = result.failed_gate
        print(f"CI FAILED at gate: {g.gate}" if g else "CI FAILED")
        return 1
    print(f"CI ERROR at stage {result.failed_stage}: {(result.error or '').strip()[:300]}")
    return 2


def cmd_git(args: argparse.Namespace) -> int:
    from core.git_sync import (
        ci_run_icon,
        git_status,
        remote_ci_status,
        status_line,
    )

    try:
        project = Project.load(args.project_id)
    except FileNotFoundError:
        print(f"ERROR: project not found: {args.project_id}", file=sys.stderr)
        return 1

    if args.git_action == "ci-status":
        status = remote_ci_status(project, limit=args.limit)
        if not status.get("available"):
            print(f"Remote CI unavailable: {status.get('error', 'unknown')}")
            return 0
        runs = status.get("runs") or []
        if not runs:
            print("No workflow runs found.")
            return 0
        for r in runs:
            state = r.get("conclusion") or r.get("status") or "?"
            print(f"  {ci_run_icon(r)} {state:12} {r.get('workflow', '?'):20} "
                  f"{r.get('branch', '?'):15} {r.get('title', '')[:50]}")
        return 0

    status = git_status(project)
    print(status_line(status))
    return 0


_DEFAULT_CATALOG = Path(__file__).resolve().parent / "packs" / "feature_catalog.datacenter_battery.yaml"


def cmd_registry(args: argparse.Namespace) -> int:
    """Source Registry: governed source-intelligence DB over DuckDB."""
    from core.acquire_bridge import (
        registry_setup,
        registry_sources_df,
        registry_status,
        registry_top_sources_df,
    )

    action = args.registry_action
    if action == "setup":
        out = registry_setup(seed=not args.no_seed, migrate=not args.no_migrate)
        print("[registry] schema ready.")
        if "seed" in out:
            s = out["seed"]
            print(f"  seeded: {s['sources']} sources, {s['platforms']} platforms, "
                  f"{s['categories']} categories, {s['inclusion_rules']} rules")
        if "migrate" in out:
            m = out["migrate"]
            print(f"  legacy migrate: {m.get('migrated', 0)} migrated, {m.get('skipped', 0)} skipped")
        counts = out["counts"]
        print(f"  totals: {counts['source_entity']} sources / "
              f"{counts['source_platform']} platforms / {counts['inclusion_rule']} rules")
        return 0

    if action == "status":
        for table, n in registry_status().items():
            print(f"  {table:28s} {n}")
        return 0

    if action == "list":
        df = registry_sources_df(active_only=not args.all)
        cols = [c for c in ["canonical_name", "source_type", "country_code",
                            "official_government", "academic_repository", "source_url"]
                if c in df.columns]
        print(df[cols].to_string(index=False) if not df.empty else "(no sources — run `mira registry setup`)")
        return 0

    if action == "top":
        df = registry_top_sources_df(limit=args.limit)
        print(df.to_string(index=False) if not df.empty else "(no scored sources yet)")
        return 0

    print(f"ERROR: unknown registry action {action!r}", file=sys.stderr)
    return 1


def cmd_macro(args: argparse.Namespace) -> int:
    """Point-in-time macro store: backfill FRED/ALFRED + ECB FX, list series."""
    # acquire/ is a sys.path root (mira convention), same as news tests/CLI.
    acquire_root = Path(__file__).resolve().parent / "acquire"
    if str(acquire_root) not in sys.path:
        sys.path.insert(0, str(acquire_root))

    from macro.ecb import DEFAULT_ECB_CURRENCIES, fetch_ecb_fx
    from macro.fred import DEFAULT_FRED_SERIES, fetch_fred_basket, month_end_vintages
    from macro.store import DEFAULT_DB, as_of, list_series, open_db, row_count, upsert_rows

    action = args.macro_action
    db_path = args.db or DEFAULT_DB

    if action == "list":
        con = open_db(db_path)
        try:
            rows = list_series(con, series_id=getattr(args, "series", None))
        finally:
            con.close()
        if not rows:
            print(f"(no macro series in {db_path} — run `mira macro backfill`)")
            return 0
        # Compact table for terminals.
        print(
            f"{'series_id':12} {'n':>7} {'first_obs':12} {'last_obs':12} "
            f"{'vintages':>8} {'source':6} {'freq':8}"
        )
        for r in rows:
            print(
                f"{str(r['series_id']):12} {int(r['n_rows']):7d} "
                f"{str(r['first_obs'])[:10]:12} {str(r['last_obs'])[:10]:12} "
                f"{int(r['n_vintages']):8d} {str(r['source'] or ''):6} "
                f"{str(r['frequency'] or ''):8}"
            )
        return 0

    if action == "asof":
        con = open_db(db_path)
        try:
            series = None
            if getattr(args, "series", None):
                series = [s.strip() for s in args.series.split(",") if s.strip()]
            rows = as_of(con, args.at, series_ids=series)
        finally:
            con.close()
        if not rows:
            print(f"(nothing available at {args.at})")
            return 0
        for r in rows:
            print(
                f"{r['series_id']:12} obs={str(r['observation_date'])[:10]} "
                f"value={r['value']} vintage={r['vintage']} "
                f"available={r['available_for_signal_at']}"
            )
        return 0

    if action == "backfill":
        source = (args.source or "fred").lower()
        series_arg = getattr(args, "series", None)
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        vintage_arg = getattr(args, "vintages", None)
        month_end = getattr(args, "month_end_vintages", False)

        vintage_dates = None
        if vintage_arg:
            vintage_dates = [v.strip() for v in vintage_arg.split(",") if v.strip()]
        elif month_end:
            if not start or not end:
                print(
                    "ERROR: --month-end-vintages requires --start and --end",
                    file=sys.stderr,
                )
                return 1
            vintage_dates = month_end_vintages(start, end)

        try:
            if source == "fred":
                ids = (
                    [s.strip() for s in series_arg.split(",") if s.strip()]
                    if series_arg
                    else list(DEFAULT_FRED_SERIES)
                )
                print(
                    f"[macro] FRED backfill series={','.join(ids)} "
                    f"vintages={len(vintage_dates) if vintage_dates else 'current'} "
                    f"window={start or '…'}..{end or '…'}"
                )
                rows = fetch_fred_basket(
                    ids,
                    vintage_dates=vintage_dates,
                    start=start,
                    end=end,
                )
            elif source == "ecb":
                ccys = (
                    [s.strip().upper() for s in series_arg.split(",") if s.strip()]
                    if series_arg
                    else list(DEFAULT_ECB_CURRENCIES)
                )
                # Allow ECB_USD or bare USD on the CLI.
                ccys = [c[4:] if c.startswith("ECB_") else c for c in ccys]
                print(
                    f"[macro] ECB FX backfill currencies={','.join(ccys)} "
                    f"window={start or '…'}..{end or '…'}"
                )
                rows = fetch_ecb_fx(ccys, start=start, end=end)
            else:
                print(
                    f"ERROR: unknown --source {source!r} (fred|ecb)",
                    file=sys.stderr,
                )
                return 1
        except Exception as exc:
            print(f"ERROR: macro fetch failed: {exc}", file=sys.stderr)
            return 1

        con = open_db(db_path)
        try:
            before = row_count(con)
            n = upsert_rows(con, rows)
            after = row_count(con)
        finally:
            con.close()
        print(
            f"[macro] upserted {n} rows "
            f"(store {before} → {after} total) → {db_path}"
        )
        return 0

    print(f"ERROR: unknown macro action {action!r}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(prog="mira", description="MIRA headless CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    acquire_p = sub.add_parser("acquire", help="Ingest a registry source into records")
    acquire_p.add_argument("--source-id", required=True)
    acquire_p.add_argument("--dry-run", action="store_true")
    acquire_p.set_defaults(func=cmd_acquire)

    reg_p = sub.add_parser("registry", help="Governed source-intelligence registry (DuckDB)")
    reg_sub = reg_p.add_subparsers(dest="registry_action", required=True)
    r_setup = reg_sub.add_parser("setup", help="Create schema + seed spine + fold in legacy registry")
    r_setup.add_argument("--no-seed", action="store_true", help="Skip curated seed")
    r_setup.add_argument("--no-migrate", action="store_true", help="Skip legacy JSON migration")
    r_setup.set_defaults(func=cmd_registry)
    r_status = reg_sub.add_parser("status", help="Row counts per registry table")
    r_status.set_defaults(func=cmd_registry)
    r_list = reg_sub.add_parser("list", help="List registered sources")
    r_list.add_argument("--all", action="store_true", help="Include inactive sources")
    r_list.set_defaults(func=cmd_registry)
    r_top = reg_sub.add_parser("top", help="Top sources by Mira Source Score")
    r_top.add_argument("--limit", type=int, default=25)
    r_top.set_defaults(func=cmd_registry)

    run_p = sub.add_parser("run", help="Run DS pipeline on a file or from records")
    run_p.add_argument("--data", default=None)
    run_p.add_argument("--from-records", action="store_true")
    run_p.add_argument("--source-id", default=None)
    run_p.add_argument("--use-existing-records", action="store_true")
    run_p.add_argument(
        "--skip-acquire-stage",
        action="store_true",
        help="After --from-records load, skip acquire pipeline stage",
    )
    run_p.add_argument("--name", default="Headless Run")
    run_p.add_argument("--task", default="regression", choices=["classification", "regression"])
    run_p.add_argument("--target", default=None)
    run_p.add_argument("--suggest-target", action="store_true")
    run_p.add_argument("--from-stage", default=None, help="Start at this pipeline stage")
    run_p.add_argument("--until", default=None, help="Stop after this stage")
    run_p.add_argument(
        "--through",
        default=None,
        help="Alias for --until (e.g. --through model)",
    )
    run_p.add_argument("--project-id", default=None)
    run_p.add_argument(
        "--split-strategy",
        default=None,
        choices=["random", "temporal"],
        help="data_prep train/test split (temporal = train oldest / test newest, no look-ahead)",
    )
    run_p.add_argument(
        "--time-column",
        default=None,
        help="raw-data column to order a temporal split by (optional; row order used if omitted)",
    )
    run_p.set_defaults(func=cmd_run)

    disc_p = sub.add_parser("discover", help="Find scrapeable sources for a topic + fields")
    disc_p.add_argument("--topic", required=True, help="Topic or prompt to search for")
    disc_p.add_argument("--fields", default="", help="Comma-separated data fields you want")
    disc_p.add_argument("--prefer", default=None, choices=["serper", "duckduckgo"],
                        help="Force a search backend (default: Serper if keyed, else DDG)")
    disc_p.add_argument("--max", type=int, default=12, help="Max candidates to return")
    disc_p.add_argument("--no-probe", action="store_true",
                        help="Skip fetching/probing candidate pages (search + score only)")
    disc_p.add_argument("--scrape", default=None,
                        help="Comma-separated candidate indices to scrape into a project")
    disc_p.add_argument("--scrape-top", type=int, default=None,
                        help="Scrape the top N candidates")
    disc_p.add_argument("--task", default="regression",
                        choices=["classification", "regression"])
    disc_p.add_argument("--through", default=None,
                        help="After scraping, run the pipeline through this stage")
    disc_p.set_defaults(func=cmd_discover)

    inv_p = sub.add_parser(
        "investigate",
        help="Goal + mode -> scrape the web -> a project with a consolidated schema")
    inv_p.add_argument("--goal", required=True,
                       help="Plain-English goal, e.g. 'investigate disc golf course difficulty'")
    inv_p.add_argument("--mode", default="explore",
                       choices=["investigate", "experiment", "predict", "explore"])
    inv_p.add_argument("--planner", default="offline",
                       choices=["offline", "ollama", "anthropic"],
                       help="Plan queries with an LLM (falls back to offline on error)")
    inv_p.add_argument("--model", default=None, help="Planner model id (for ollama/anthropic)")
    inv_p.add_argument("--prefer", default=None, choices=["serper", "duckduckgo", "mojeek"],
                       help="Force a search backend (default: Serper if keyed, else keyless)")
    inv_p.add_argument("--target-n", type=int, default=100,
                       help="Max candidates to gather from search (default 100)")
    inv_p.add_argument("--max-scrape", type=int, default=60,
                       help="Max candidates to actually scrape (default 60)")
    inv_p.add_argument("--time-budget", type=float, default=180.0,
                       help="Stop scraping after this many seconds (default 180)")
    inv_p.add_argument("--no-create", action="store_true",
                       help="Consolidate a schema but don't create a project")
    inv_p.add_argument("--json", action="store_true", help="Emit the summary as JSON")
    inv_p.set_defaults(func=cmd_investigate)

    board_p = sub.add_parser("board", help="Per-project Kanban board")
    board_sub = board_p.add_subparsers(dest="board_action", required=True)

    b_list = board_sub.add_parser("list", help="Show the board")
    b_list.add_argument("--project-id", required=True)
    b_list.add_argument("--tasks-only", action="store_true",
                        help="Hide derived pipeline-stage cards")
    b_list.set_defaults(func=cmd_board)

    b_add = board_sub.add_parser("add", help="Add a task card")
    b_add.add_argument("--project-id", required=True)
    b_add.add_argument("--title", required=True)
    b_add.add_argument("--description", default=None)
    b_add.add_argument("--column", default="backlog")
    b_add.add_argument("--priority", default="medium",
                       choices=["low", "medium", "high", "urgent"])
    b_add.add_argument("--points", type=int, default=None,
                       choices=[1, 2, 3, 5, 8, 13, 21])
    b_add.add_argument("--assignee", default=None)
    b_add.add_argument("--labels", default="", help="Comma-separated labels")
    b_add.set_defaults(func=cmd_board)

    b_move = board_sub.add_parser("move", help="Move a task to a column")
    b_move.add_argument("--project-id", required=True)
    b_move.add_argument("--task", type=int, required=True)
    b_move.add_argument("--to", required=True, help="Target column")
    b_move.set_defaults(func=cmd_board)

    b_run = board_sub.add_parser("run", help="Execute a task via its bound skill")
    b_run.add_argument("--project-id", required=True)
    b_run.add_argument("--task", type=int, required=True)
    b_run.add_argument("--provider", default=None,
                       choices=["offline", "ollama", "anthropic"])
    b_run.add_argument("--dry-run", action="store_true",
                       help="Resolve context + report what would run, without running")
    b_run.add_argument("--show-output", action="store_true",
                       help="Print the full execution output")
    b_run.set_defaults(func=cmd_board)

    b_ctx = board_sub.add_parser("context", help="Show/assign a task's context")
    b_ctx.add_argument("--project-id", required=True)
    b_ctx.add_argument("--task", type=int, required=True)
    b_ctx.add_argument("--refresh", action="store_true",
                       help="Re-resolve context live and persist the snapshot")
    b_ctx.add_argument("--brief", action="store_true",
                       help="Print the markdown brief for a human/agent executor")
    b_ctx.set_defaults(func=cmd_board)

    data_p = sub.add_parser("data", help="Dataset analysis for preprocessing")
    data_sub = data_p.add_subparsers(dest="data_action", required=True)
    d_an = data_sub.add_parser("analyze", help="Analyze the project's working data")
    d_an.add_argument("--project-id", required=True)
    d_an.add_argument("--full", action="store_true", help="Full per-column analysis")
    d_an.add_argument("--json", action="store_true", help="Emit JSON")
    d_an.add_argument("--force", action="store_true", help="Bypass the cache")
    d_an.set_defaults(func=cmd_data)

    d_ex = data_sub.add_parser("explore", help="Notebook-style concept analyses")
    d_ex.add_argument("--project-id", required=True)
    d_ex.add_argument("--run", default=None, help="Execute one recipe by key")
    d_ex.add_argument("--notebook", action="store_true",
                      help="Write exports/exploration.ipynb")
    d_ex.set_defaults(func=cmd_data)

    sprint_p = sub.add_parser("sprint", help="Sprint planning workflow")
    sprint_sub = sprint_p.add_subparsers(dest="sprint_action", required=True)
    s_plan = sprint_sub.add_parser("plan", help="Generate a sprint plan")
    s_plan.add_argument("--project-id", required=True)
    s_plan.add_argument("--capacity", type=int, default=None,
                        help="Story-point budget for the sprint")
    s_plan.add_argument("--provider", default=None,
                        choices=["offline", "ollama", "anthropic"],
                        help="LLM provider (default: project meta llm.provider)")
    s_plan.add_argument("--export", default=None, choices=["md", "json"],
                        help="Print the plan as markdown or JSON")
    s_plan.add_argument("--materialize", action="store_true",
                        help="Create board tasks from the generated stories")
    s_plan.set_defaults(func=cmd_sprint)

    ci_p = sub.add_parser("ci", help="Gated CI pipeline run")
    ci_sub = ci_p.add_subparsers(dest="ci_action", required=True)
    c_run = ci_sub.add_parser("run", help="Run the gated pipeline")
    c_run.add_argument("--project-id", required=True)
    c_run.add_argument("--from-stage", default=None, help="Start at this stage")
    c_run.add_argument("--through", default=None, help="Stop after this stage")
    c_run.set_defaults(func=cmd_ci)

    git_p = sub.add_parser("git", help="Per-project git status")
    git_sub = git_p.add_subparsers(dest="git_action", required=True)
    g_status = git_sub.add_parser("status", help="Show repo status")
    g_status.add_argument("--project-id", required=True)
    g_status.set_defaults(func=cmd_git)

    g_ci = git_sub.add_parser("ci-status", help="Show recent GitHub Actions runs")
    g_ci.add_argument("--project-id", required=True)
    g_ci.add_argument("--limit", type=int, default=5)
    g_ci.set_defaults(func=cmd_git)

    macro_p = sub.add_parser(
        "macro",
        help="Point-in-time macro store (FRED/ALFRED + ECB FX)",
    )
    macro_sub = macro_p.add_subparsers(dest="macro_action", required=True)

    m_back = macro_sub.add_parser(
        "backfill",
        help="Fetch series and upsert into the MIRA macro DuckDB",
    )
    m_back.add_argument(
        "--series",
        default=None,
        help="Comma-separated series ids (FRED: VIXCLS,DGS10; ECB: USD,GBP). "
             "Default = built-in basket for --source",
    )
    m_back.add_argument(
        "--source",
        default="fred",
        choices=["fred", "ecb"],
        help="Adapter to run (default fred)",
    )
    m_back.add_argument("--start", default=None, help="Observation window start YYYY-MM-DD")
    m_back.add_argument("--end", default=None, help="Observation window end YYYY-MM-DD")
    m_back.add_argument(
        "--vintages",
        default=None,
        help="Comma-separated ALFRED vintage dates (fred only); "
             "stamps available_for_signal_at per vintage",
    )
    m_back.add_argument(
        "--month-end-vintages",
        action="store_true",
        help="ALFRED: one vintage per month-end in --start..--end (fred only)",
    )
    m_back.add_argument(
        "--db",
        default=None,
        help="DuckDB path (default var/macro/macro_series.duckdb)",
    )
    m_back.set_defaults(func=cmd_macro)

    m_list = macro_sub.add_parser("list", help="List series in the macro store")
    m_list.add_argument(
        "--series",
        default=None,
        help="Filter to one series_id",
    )
    m_list.add_argument(
        "--db",
        default=None,
        help="DuckDB path (default var/macro/macro_series.duckdb)",
    )
    m_list.set_defaults(func=cmd_macro)

    m_asof = macro_sub.add_parser(
        "asof",
        help="Point-in-time latest known value per series",
    )
    m_asof.add_argument(
        "--at",
        required=True,
        help="As-of timestamp (YYYY-MM-DD or full ISO)",
    )
    m_asof.add_argument(
        "--series",
        default=None,
        help="Comma-separated series filter",
    )
    m_asof.add_argument(
        "--db",
        default=None,
        help="DuckDB path (default var/macro/macro_series.duckdb)",
    )
    m_asof.set_defaults(func=cmd_macro)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())