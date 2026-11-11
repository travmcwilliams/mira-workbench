"""Generate downloadable PDF analysis reports from tabular data."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from core.data_loader import detect_schema, load_file, suggest_target
from core.dataset_profiles import detect_dataset_profile, profile_label
from core.restaurant_features import engineer_labor_features
from core.server_analytics import prepare_server_cohort, top_vs_bottom_summary


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.92, title, fontsize=16, fontweight="bold", va="top")
    y = 0.86
    for line in lines:
        fig.text(0.08, y, line, fontsize=10, va="top", family="monospace")
        y -= 0.035
        if y < 0.06:
            break
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None


def _daily_ops_stats(df: pd.DataFrame) -> dict[str, Any]:
    date_col = _resolve_col(df, "trading_day_date")
    sales_col = _resolve_col(df, "net_sales")
    tips_col = _resolve_col(df, "tips")
    labor_pct_col = _resolve_col(df, "labor_pct")
    splh_col = _resolve_col(df, "sales_per_labor_hour")
    dow_col = _resolve_col(df, "day_of_week")

    working = df.copy()
    if date_col:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    if sales_col:
        working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
        active = working[working[sales_col] > 0]
    else:
        active = working

    stats: dict[str, Any] = {
        "total_days": len(df),
        "active_days": len(active),
    }
    if labor_pct_col and len(active):
        stats["labor_pct_mean"] = float(active[labor_pct_col].mean())
        stats["labor_pct_median"] = float(active[labor_pct_col].median())
    if tips_col and len(active):
        stats["tips_mean"] = float(active[tips_col].mean())
        stats["tips_total"] = float(active[tips_col].sum())
    if sales_col and len(active):
        stats["sales_mean"] = float(active[sales_col].mean())
        stats["sales_total"] = float(active[sales_col].sum())
    if splh_col and len(active):
        stats["splh_mean"] = float(active[splh_col].mean())

    if dow_col and labor_pct_col and len(active):
        by_dow = (
            active.groupby(dow_col, observed=True)[labor_pct_col]
            .mean()
            .sort_values()
        )
        stats["best_dow"] = str(by_dow.index[0])
        stats["best_dow_labor_pct"] = float(by_dow.iloc[0])
        stats["worst_dow"] = str(by_dow.index[-1])
        stats["worst_dow_labor_pct"] = float(by_dow.iloc[-1])

    return stats


def build_restaurant_daily_pdf(df: pd.DataFrame, out_path: Path) -> Path:
    """EDA + ops summary PDF for daily restaurant data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = _daily_ops_stats(df)

    date_col = _resolve_col(df, "trading_day_date")
    sales_col = _resolve_col(df, "net_sales")
    tips_col = _resolve_col(df, "tips")
    labor_pct_col = _resolve_col(df, "labor_pct")
    splh_col = _resolve_col(df, "sales_per_labor_hour")
    dow_col = _resolve_col(df, "day_of_week")
    labor_hours_col = _resolve_col(df, "labor_hours")

    working = df.copy()
    if date_col:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working = working.dropna(subset=[date_col]).sort_values(date_col)
    if sales_col:
        working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
        active = working[working[sales_col] > 0].copy()
    else:
        active = working.copy()

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary_lines = [
        f"Generated: {generated}",
        f"Rows: {stats.get('total_days', 0):,} total | {stats.get('active_days', 0):,} with sales > 0",
        "",
        "KEY METRICS (days with sales)",
    ]
    if "labor_pct_mean" in stats:
        summary_lines.append(
            f"  Labor %     mean {stats['labor_pct_mean']:.1%}  |  median {stats['labor_pct_median']:.1%}"
        )
    if "tips_mean" in stats:
        summary_lines.append(
            f"  Tips/day    mean ${stats['tips_mean']:,.0f}  |  total ${stats['tips_total']:,.0f}"
        )
    if "sales_mean" in stats:
        summary_lines.append(
            f"  Sales/day   mean ${stats['sales_mean']:,.0f}  |  total ${stats['sales_total']:,.0f}"
        )
    if "splh_mean" in stats:
        summary_lines.append(f"  Sales/labor hr (avg): ${stats['splh_mean']:,.1f}")
    if "best_dow" in stats:
        summary_lines.extend(
            [
                "",
                "DAY-OF-WEEK LABOR %",
                f"  Best:  {stats['best_dow']} ({stats['best_dow_labor_pct']:.1%})",
                f"  Worst: {stats['worst_dow']} ({stats['worst_dow_labor_pct']:.1%})",
            ]
        )
    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "  - Labor % = wages / net_sales; lower is better on high-volume days.",
            "  - Weekend days (Fri/Sat) typically show better sales leverage.",
            "  - Monday/Tuesday often run higher labor % with softer revenue.",
        ]
    )

    with PdfPages(out_path) as pdf:
        _text_page(pdf, "Restaurant — Daily Ops Analysis", summary_lines)

        if dow_col and labor_pct_col and len(active):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            plot_df = active.copy()
            plot_df[dow_col] = pd.Categorical(plot_df[dow_col], categories=order, ordered=True)
            sns.barplot(
                data=plot_df.groupby(dow_col, observed=True)[labor_pct_col].mean().reset_index(),
                x=dow_col,
                y=labor_pct_col,
                ax=ax,
                color="#2E86AB",
            )
            ax.set_title("Mean labor cost % by day of week")
            ax.set_ylabel("Labor %")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            plt.xticks(rotation=30, ha="right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if date_col and tips_col and len(active):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.plot(active[date_col], active[tips_col], color="#A23B72", linewidth=1)
            ax.set_title("Daily tips trend")
            ax.set_ylabel("Tips ($)")
            ax.set_xlabel("Date")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if date_col and sales_col and len(active):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.plot(active[date_col], active[sales_col], color="#F18F01", linewidth=1)
            ax.set_title("Daily net sales trend")
            ax.set_ylabel("Net sales ($)")
            ax.set_xlabel("Date")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if date_col and splh_col and len(active):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.plot(active[date_col], active[splh_col], color="#3B1F2B", linewidth=1)
            ax.set_title("Sales per labor hour over time")
            ax.set_ylabel("$/labor hr")
            ax.set_xlabel("Date")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if labor_hours_col and sales_col and len(active):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            sc = ax.scatter(
                active[labor_hours_col],
                active[sales_col],
                c=active[labor_pct_col] if labor_pct_col else "#2E86AB",
                alpha=0.6,
                cmap="viridis",
            )
            ax.set_xlabel("Labor hours")
            ax.set_ylabel("Net sales ($)")
            ax.set_title("Labor hours vs net sales")
            if labor_pct_col:
                plt.colorbar(sc, ax=ax, label="Labor %")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        numeric = active.select_dtypes(include=[np.number]).columns.tolist()
        if labor_pct_col and labor_pct_col in numeric:
            numeric = [c for c in numeric if c != labor_pct_col]
        if len(numeric) >= 2:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            corr = active[numeric + ([labor_pct_col] if labor_pct_col else [])].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Feature correlations (numeric)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_path


def build_employee_labor_pdf(df: pd.DataFrame, out_path: Path) -> Path:
    """PDF for employee/role-level labor exports with engineered features."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_level, _ = engineer_labor_features(
        df, target=None, add_peer_benchmarks=True, aggregate_employee=False
    )
    emp_level, _ = engineer_labor_features(
        df, target=None, add_peer_benchmarks=True, aggregate_employee=True
    )

    tips_col = _resolve_col(row_level, "total_tips")
    hours_col = _resolve_col(row_level, "total_hours")
    sales_col = _resolve_col(row_level, "net_sales")
    pay_col = _resolve_col(row_level, "total_pay")
    role_col = _resolve_col(row_level, "role")

    lines = [
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Rows: {len(row_level):,} employee-role | {len(emp_level):,} employees",
        "",
    ]
    if tips_col:
        lines.append(f"Total tips:      ${row_level[tips_col].sum():,.0f}")
    if hours_col:
        lines.append(f"Total hours:     {row_level[hours_col].sum():,.0f}")
    if sales_col:
        lines.append(f"Attributed sales: ${row_level[sales_col].sum():,.0f}")
    if pay_col:
        lines.append(f"Total pay:       ${row_level[pay_col].sum():,.0f}")
    if "is_multi_role" in row_level.columns and "employee_key" in row_level.columns:
        multi = int(row_level.groupby("employee_key", observed=True)["is_multi_role"].max().sum())
        lines.append(f"Multi-role staff: {multi:,} employees")
    if "implied_shifts" in row_level.columns:
        lines.append(f"Implied shifts:  {row_level['implied_shifts'].sum():,.0f} (@ 8 hr)")

    with PdfPages(out_path) as pdf:
        _text_page(pdf, "Restaurant — Employee Labor Analysis", lines)

        if role_col and tips_col:
            by_role = row_level.groupby(role_col, observed=True)[tips_col].sum().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8.5, max(4, len(by_role) * 0.25)))
            by_role.plot(kind="barh", ax=ax, color="#2E86AB")
            ax.set_title("Total tips by role")
            ax.set_xlabel("Tips ($)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if hours_col and tips_col:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.scatter(row_level[hours_col], row_level[tips_col], alpha=0.5, color="#A23B72")
            ax.set_xlabel("Total hours")
            ax.set_ylabel("Tips ($)")
            ax.set_title("Hours vs tips (role rows)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if "tips_per_hour_vs_role_median" in row_level.columns and role_col:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            for role_name, grp in row_level.groupby(role_col, observed=True):
                ax.scatter(
                    grp["tips_per_hour_vs_role_median"],
                    grp[tips_col],
                    alpha=0.6,
                    label=str(role_name),
                )
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Tips/hour vs role median")
            ax.set_ylabel("Tips ($)")
            ax.set_title("Peer benchmark: tip productivity")
            ax.legend(fontsize=7, loc="best")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if "n_roles" in emp_level.columns and tips_col:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.scatter(emp_level["n_roles"], emp_level[tips_col], alpha=0.6, color="#F18F01")
            ax.set_xlabel("Roles held")
            ax.set_ylabel("Total tips ($)")
            ax.set_title("Employee aggregation: roles vs total tips")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        share_cols = [
            c for c in emp_level.columns if c.startswith("employee_") and c.endswith("_hours_share")
        ]
        if share_cols and len(emp_level) >= 3:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            top = emp_level.nlargest(min(12, len(emp_level)), tips_col)
            bottom = np.arange(len(top))
            left = np.zeros(len(top))
            for col in share_cols:
                label = col.replace("employee_", "").replace("_hours_share", "")
                vals = top[col].fillna(0).values
                ax.barh(bottom, vals, left=left, label=label)
                left += vals
            ax.set_yticks(bottom)
            ax.set_yticklabels(top.get("employee_key", top.index.astype(str)).astype(str), fontsize=8)
            ax.set_xlabel("Hours share")
            ax.set_title("Segment mix — top earners")
            ax.legend(fontsize=7, loc="lower right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        servers, srv_meta = prepare_server_cohort(df, min_hours=20.0)
        if not servers.empty and "tips_per_hour" in servers.columns:
            srv_summary = top_vs_bottom_summary(servers, metric="tips_per_hour", n=3)
            srv_lines = [
                "SERVER COHORT (role = Server, min 20 hrs)",
                f"  Servers: {srv_meta.get('server_count', 0)}",
                f"  Median tips/hr: ${srv_summary.get('cohort_median', 0):,.2f}",
                f"  Top 3 avg vs median: +{srv_summary.get('top_pct_above_median', 0):.0f}%",
                "",
                "TOP SERVERS (tips/hr)",
            ]
            pct_map = servers.set_index("display_name")["tips_per_hour_pct_above_median"]
            for name, val in zip(
                srv_summary.get("top_names", []),
                srv_summary.get("top_values", []),
            ):
                pct = float(pct_map.get(name, 0))
                srv_lines.append(f"  {name}: ${val:,.2f}/hr ({pct:+.0f}% vs median)")
            _text_page(pdf, "Server Performance Summary", srv_lines)

            fig, ax = plt.subplots(figsize=(8.5, max(4, len(servers) * 0.3)))
            ordered = servers.sort_values("tips_per_hour", ascending=True)
            ax.boxplot(
                ordered["tips_per_hour"],
                vert=False,
                widths=0.4,
                patch_artist=True,
                boxprops={"facecolor": "#2E86AB", "alpha": 0.4},
                medianprops={"color": "#F18F01", "linewidth": 2},
            )
            ax.scatter(
                ordered["tips_per_hour"],
                np.ones(len(ordered)),
                alpha=0.7,
                color="#A23B72",
                s=40,
                zorder=3,
            )
            for _, row in ordered.iterrows():
                ax.annotate(
                    row["display_name"],
                    (row["tips_per_hour"], 1.02),
                    fontsize=7,
                    ha="center",
                )
            ax.set_xlabel("Tips per hour ($)")
            ax.set_title("Server tips/hr — whisker plot with labels")
            ax.set_yticks([])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if "tips_per_hour_pct_above_median" in servers.columns:
                fig, ax = plt.subplots(figsize=(8.5, max(4, len(servers) * 0.28)))
                pct_ord = servers.sort_values("tips_per_hour_pct_above_median", ascending=True)
                colors = ["#2E86AB" if v >= 0 else "#A23B72" for v in pct_ord["tips_per_hour_pct_above_median"]]
                ax.barh(
                    pct_ord["display_name"],
                    pct_ord["tips_per_hour_pct_above_median"],
                    color=colors,
                )
                ax.axvline(0, color="gray", linewidth=1)
                ax.set_xlabel("% above/below server median")
                ax.set_title("How much better — tips per hour vs cohort")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return out_path


def build_modeling_summary_pdf(
    df: pd.DataFrame,
    out_path: Path,
    *,
    target: str,
    task: str,
    metrics: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> Path:
    """Short PDF summarizing predictive modeling results."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = detect_schema(df)
    suggestion = suggest_target(df, schema, task_type=task)

    lines = [
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Task: {task}",
        f"Target: {target}",
        f"Rows: {len(df):,}  |  Columns: {len(df.columns)}",
        "",
        "MODEL METRICS",
    ]
    if metrics:
        for key, val in metrics.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")
    else:
        lines.append("  (Run modeling pipeline to populate metrics)")

    if feature_names:
        lines.extend(["", "TOP FEATURES", *[f"  - {f}" for f in feature_names[:12]]])

    lines.extend(
        [
            "",
            "NOTES",
            "  - labor_pct models exclude wage/sales inputs to avoid target leakage.",
            "  - Use daily dashboard for trends; employee PDF for role mix.",
        ]
    )

    with PdfPages(out_path) as pdf:
        _text_page(pdf, "Modeling Summary", lines)
    return out_path


def generate_analysis_pdfs(
    data_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "analysis",
) -> list[Path]:
    """Auto-detect dataset profile and write appropriate PDF reports."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_file(data_path)
    profile = detect_dataset_profile(df)
    stamp = datetime.now().strftime("%Y%m%d")
    created: list[Path] = []

    if profile == "restaurant_daily":
        p = build_restaurant_daily_pdf(
            df, output_dir / f"{prefix}_daily_ops_{stamp}.pdf"
        )
        created.append(p)
        target = suggest_target(df, detect_schema(df), "regression").get("target_column") or "labor_pct"
        created.append(
            build_modeling_summary_pdf(
                df,
                output_dir / f"{prefix}_modeling_brief_{stamp}.pdf",
                target=target,
                task="regression",
            )
        )
    elif profile == "employee_labor":
        created.append(
            build_employee_labor_pdf(df, output_dir / f"{prefix}_employee_labor_{stamp}.pdf")
        )
    else:
        stats = df.describe().transpose()
        lines = [
            f"Profile: {profile_label(profile)}",
            f"Rows: {len(df):,}",
            "",
            "Numeric summary (top 8 columns):",
        ]
        for col in stats.head(8).index:
            mean = stats.loc[col, "mean"]
            lines.append(f"  {col}: mean={mean:.4g}")
        out = output_dir / f"{prefix}_generic_{stamp}.pdf"
        with PdfPages(out) as pdf:
            _text_page(pdf, "Generic Data Summary", lines)
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(8.5, 6))
                sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlations")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        created.append(out)

    return created