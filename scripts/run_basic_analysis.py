"""Run basic DS analysis and export PDF reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis_runner import run_full_pipeline
from core.data_loader import detect_schema, load_file, suggest_target
from core.pdf_report import build_modeling_summary_pdf, generate_analysis_pdfs


def main() -> int:
    parser = argparse.ArgumentParser(description="Basic DS analysis + PDF export")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "sample" / "restaurant_daily_sample.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports"),
        help="Directory for PDF output",
    )
    parser.add_argument("--run-model", action="store_true", help="Run full ML pipeline")
    parser.add_argument("--target", default=None)
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        return 1

    df = load_file(data_path)
    schema = detect_schema(df)
    target = args.target
    if not target:
        target = suggest_target(df, schema, args.task).get("target_column")
    if not target:
        print("ERROR: could not determine target column", file=sys.stderr)
        return 1

    print(f"Data: {data_path.name} ({len(df)} rows)")
    print(f"Task: {args.task} | Target: {target}")

    pdfs = generate_analysis_pdfs(data_path, output_dir, prefix=data_path.stem)
    print(f"\nPDF reports ({len(pdfs)}):")
    for p in pdfs:
        print(f"  {p.resolve()}")

    if args.run_model:
        print("\nRunning ML pipeline...")
        project, metrics = run_full_pipeline(data_path, target, args.task)
        model_pdf = build_modeling_summary_pdf(
            df,
            output_dir / f"{data_path.stem}_model_results.pdf",
            target=target,
            task=args.task,
            metrics=metrics,
        )
        pdfs.append(model_pdf)
        print(f"  {model_pdf.resolve()}")
        print(f"Project artifacts: {project.root.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())