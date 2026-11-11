"""CLI: export server performance charts to reports/ and Downloads."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.chart_export import export_server_chart_pack
from core.data_loader import load_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export server performance chart pack")
    parser.add_argument(
        "--data",
        default=str(root / "data" / "sample" / "restaurant_labor_sample.csv"),
    )
    parser.add_argument("--output", default=str(root / "reports"))
    parser.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    parser.add_argument("--min-hours", type=float, default=20.0)
    parser.add_argument("--metric", default="tips_per_hour")
    args = parser.parse_args()

    df = load_file(args.data)
    result = export_server_chart_pack(
        df,
        args.output,
        min_hours=args.min_hours,
        rank_metric=args.metric,
        prefix="restaurant_servers",
    )
    if result.get("error"):
        raise SystemExit(result["error"])

    downloads = Path(args.downloads)
    downloads.mkdir(parents=True, exist_ok=True)
    dest_zip = downloads / result["zip_path"].name
    shutil.copy2(result["zip_path"], dest_zip)

    print(f"Charts: {len(result['figures'])}")
    print(f"ZIP: {result['zip_path']}")
    print(f"Copied to: {dest_zip}")
    if not result.get("png_export"):
        print("Note: PNG export skipped — pip install kaleido for PNG files in the ZIP")


if __name__ == "__main__":
    main()