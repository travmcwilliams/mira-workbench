"""Export interactive charts to HTML, PNG, CSV, and ZIP bundles."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.server_analytics import prepare_server_cohort, server_leaderboard, top_vs_bottom_summary
from core.server_charts import build_server_figures

_PNG_SCALE = 2


def _write_png(fig: Any, path: Path) -> bool:
    try:
        fig.write_image(str(path), scale=_PNG_SCALE, width=1200, height=700)
        return True
    except Exception:
        pass
    try:
        import kaleido  # noqa: F401

        from kaleido.scopes.plotly import PlotlyScope

        scope = PlotlyScope()
        img = scope.transform(fig, format="png", width=1200, height=700, scale=_PNG_SCALE)
        path.write_bytes(img)
        return True
    except Exception:
        return False


def export_server_chart_pack(
    df: pd.DataFrame,
    output_dir: Path | str,
    *,
    min_hours: float = 20.0,
    rank_metric: str = "tips_per_hour",
    prefix: str = "server_charts",
) -> dict[str, Any]:
    """
    Write server performance charts + leaderboard to output_dir.
    Returns paths and zip bytes for Streamlit download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    servers, meta = prepare_server_cohort(df, min_hours=min_hours)
    if servers.empty:
        return {"error": "No server rows", "files": []}

    summary = top_vs_bottom_summary(servers, metric=rank_metric, n=3)
    med = summary.get("cohort_median")
    figures = build_server_figures(servers, rank_metric=rank_metric, cohort_median=med)

    pack_dir = output_dir / f"{prefix}_{stamp}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    png_ok = False

    # Leaderboard CSV
    board = server_leaderboard(servers, metric=rank_metric)
    csv_path = pack_dir / "server_leaderboard.csv"
    board.to_csv(csv_path, index=False)
    written.append(csv_path)

    # Summary text
    summary_path = pack_dir / "README.txt"
    lines = [
        "Restaurant — Server Performance Chart Pack",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Servers: {meta.get('server_count', 0)} (min {min_hours} hrs)",
        f"Rank metric: {rank_metric}",
        f"Cohort median: {med}",
        "",
        "Top 3:",
    ]
    for name, val in zip(summary.get("top_names", []), summary.get("top_values", [])):
        lines.append(f"  {name}: {val}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    written.append(summary_path)

    html_dir = pack_dir / "html"
    png_dir = pack_dir / "png"
    html_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)

    for name, fig in figures.items():
        html_path = html_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        written.append(html_path)
        png_path = png_dir / f"{name}.png"
        if _write_png(fig, png_path):
            written.append(png_path)
            png_ok = True

    # Combined HTML dashboard (all charts stacked)
    combined_path = pack_dir / "all_charts.html"
    sections = [
        f'<h2>{name.replace("_", " ").title()}</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}'
        for name, fig in figures.items()
    ]
    combined_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Server Charts</title>"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
        "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:1rem}</style>"
        "</head><body><h1>Server Performance Charts</h1>"
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    written.append(combined_path)

    zip_path = output_dir / f"{prefix}_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in written:
            arcname = fpath.relative_to(pack_dir.parent)
            zf.write(fpath, arcname)

    zip_bytes = zip_path.read_bytes()

    return {
        "pack_dir": pack_dir,
        "zip_path": zip_path,
        "zip_bytes": zip_bytes,
        "files": written,
        "png_export": png_ok,
        "server_count": meta.get("server_count", 0),
        "figures": list(figures.keys()),
    }