"""Build the widest MIRA stock panel from a SQLite daily-bars market.db.

Includes base technical features, daily/intraday market metrics, macro as-of join,
stock_feature_snapshots, and sentiment/catalyst aggregates (NaN + flags when empty).

Outputs Parquet (primary) plus a JSON manifest with column groups and stats.

Usage:
    python scripts/build_stock_panel_full.py --tickers 1000
    python scripts/build_stock_panel_full.py --tickers 500 --csv
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_stock_panel import (  # noqa: E402
    DEFAULT_DB,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    compute_ticker_features,
    load_bars,
    load_top_tickers,
)

DAILY_METRICS = {
    "mkt_daily_return_1d": "mmo_ret_1d",
    "mkt_daily_return_5d": "mmo_ret_5d",
    "mkt_daily_rsi_14": "mmo_rsi_14",
    "mkt_daily_sma_20": "mmo_sma_20",
    "mkt_daily_sma_50": "mmo_sma_50",
    "mkt_daily_volatility_20d": "mmo_vol_20d",
}

INTRADAY_METRICS = {
    "mkt_intraday_vwap_deviation": "vwap_dev",
    "mkt_intraday_deviation_zscore": "dev_z",
    "mkt_intraday_spread_proxy": "spread",
}

MACRO_SERIES = ["VIXCLS", "DGS10", "T10Y2Y", "DFF", "DEXUSEU", "UNRATE", "CPIAUCSL"]

CATALYST_METRICS = {
    "ctx_catalyst_sentiment": "cat_sentiment",
    "ctx_catalyst_relevance": "cat_relevance",
    "ctx_catalyst_materiality": "cat_materiality",
    "ctx_catalyst_confidence": "cat_confidence",
}

SENTIMENT_PLACEHOLDER_COLS = [
    "news_tone",
    "news_volume",
    "cat_doc_count",
    *CATALYST_METRICS.values(),
]

SNAPSHOT_SKIP = {
    "snapshot_id",
    "symbol_id",
    "ticker",
    "snapshot_ts",
    "snapshot_date",
    "universe_key",
    "data_version",
    "source_hashes_json",
    "availability_json",
    "quality_flags_json",
    "created_at_utc",
    "rates_regime_label",
}


def _snapshot_export_cols(con: sqlite3.Connection) -> list[str]:
    cols = con.execute("PRAGMA table_info(stock_feature_snapshots)").fetchall()
    return [c[1] for c in cols if c[1] not in SNAPSHOT_SKIP and c[2] in ("REAL", "INTEGER")]


def build_base_panel(
    db_path: Path,
    *,
    n_tickers: int,
    fwd_days: int = 5,
    min_rows_per_ticker: int = 252,
) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        tickers = load_top_tickers(con, n_tickers)
        if not tickers:
            raise SystemExit("No tickers found. Check market.db path and universe_memberships.")

        print(f"Building base panel for {len(tickers)} tickers")
        frames: list[pd.DataFrame] = []

        for i, (ticker, sector, symbol_id) in enumerate(tickers, 1):
            bars = load_bars(con, symbol_id)
            if len(bars) < min_rows_per_ticker:
                continue

            feat = compute_ticker_features(bars, fwd_days=fwd_days)
            feat["ticker"] = ticker
            feat["sector"] = sector
            feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.strftime("%Y-%m-%d")

            out = feat[OUTPUT_COLUMNS].copy()
            out = out.dropna(subset=["ret_1d", "ret_5d", "rsi_14", "fwd_ret_5d"])
            if out.empty:
                continue
            frames.append(out)

            if i % 50 == 0 or i == len(tickers):
                print(f"  base: {i}/{len(tickers)} tickers, rows: {sum(len(f) for f in frames):,}")

        if not frames:
            raise SystemExit("No feature rows produced.")

        panel = pd.concat(frames, ignore_index=True)
        return panel.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    finally:
        con.close()


def _load_daily_metrics_bulk(con: sqlite3.Connection, tickers: set[str]) -> pd.DataFrame:
    ph_metrics = ",".join("?" * len(DAILY_METRICS))
    ph_tickers = ",".join("?" * len(tickers))
    sql = f"""
        WITH ranked AS (
            SELECT ticker,
                   DATE(event_time) AS trade_date,
                   metric_id,
                   metric_value,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, DATE(event_time), metric_id
                       ORDER BY event_time DESC
                   ) AS rn
            FROM market_metric_observations
            WHERE metric_id IN ({ph_metrics})
              AND ticker IN ({ph_tickers})
        )
        SELECT ticker, trade_date, metric_id, metric_value
        FROM ranked
        WHERE rn = 1
    """
    params = [*DAILY_METRICS.keys(), *sorted(tickers)]
    df = pd.read_sql_query(sql, con, params=params)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    wide = (
        df.pivot_table(index=["ticker", "trade_date"], columns="metric_id", values="metric_value", aggfunc="last")
        .rename(columns=DAILY_METRICS)
        .reset_index()
    )
    return wide


def _load_intraday_daily_bulk(con: sqlite3.Connection, tickers: set[str]) -> pd.DataFrame:
    ph_metrics = ",".join("?" * len(INTRADAY_METRICS))
    ph_tickers = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker, event_time, metric_id, metric_value
        FROM market_metric_observations
        WHERE metric_id IN ({ph_metrics})
          AND ticker IN ({ph_tickers})
    """
    print("  loading intraday observations (bulk)…")
    df = pd.read_sql_query(sql, con, params=[*INTRADAY_METRICS.keys(), *sorted(tickers)])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    df["trade_date"] = pd.to_datetime(df["event_time"]).dt.strftime("%Y-%m-%d")
    df["m"] = df["metric_id"].map(INTRADAY_METRICS)
    g = (
        df.groupby(["ticker", "trade_date", "m"], observed=True)["metric_value"]
        .agg(["mean", "last", "std"])
        .unstack("m")
    )
    g.columns = [f"intra_{m}_{stat}" for stat, m in g.columns]
    return g.reset_index()


def _macro_by_date(con: sqlite3.Connection, dates: pd.Series) -> pd.DataFrame:
    macro_raw = pd.read_sql_query(
        "SELECT series_id, available_for_signal_at, value FROM macro_observations",
        con,
    )
    if macro_raw.empty:
        return pd.DataFrame({"trade_date": pd.unique(dates)})

    macro_raw["asof"] = pd.to_datetime(macro_raw["available_for_signal_at"]).dt.tz_localize(None)
    base = pd.DataFrame({"trade_date": pd.unique(dates)})
    base["dt"] = pd.to_datetime(base["trade_date"])
    base = base.sort_values("dt")

    for series in MACRO_SERIES:
        sub = (
            macro_raw.loc[macro_raw["series_id"] == series, ["asof", "value"]]
            .dropna()
            .sort_values("asof")
        )
        col = f"macro_{series.lower()}"
        if sub.empty:
            base[col] = np.nan
            continue
        merged = pd.merge_asof(base, sub, left_on="dt", right_on="asof", direction="backward")
        base[col] = merged["value"].values

    return base.drop(columns=["dt"])


def _load_snapshots(con: sqlite3.Connection, tickers: set[str]) -> pd.DataFrame:
    snap_cols = _snapshot_export_cols(con)
    if not snap_cols:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    col_sql = ", ".join(f'"{c}"' for c in snap_cols)
    ph_tickers = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker, snapshot_date AS trade_date, {col_sql}
        FROM stock_feature_snapshots
        WHERE ticker IN ({ph_tickers})
    """
    try:
        df = pd.read_sql_query(sql, con, params=sorted(tickers))
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    if df.empty:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    rename = {c: f"snap_{c}" for c in snap_cols}
    return df.rename(columns=rename)


def _load_catalyst_daily(con: sqlite3.Connection, tickers: set[str]) -> pd.DataFrame:
    try:
        n = con.execute("SELECT COUNT(*) FROM context_metric_observations").fetchone()[0]
    except sqlite3.OperationalError:
        n = 0

    if n == 0:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    ph_metrics = ",".join("?" * len(CATALYST_METRICS))
    ph_tickers = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker,
               DATE(event_time) AS trade_date,
               metric_id,
               AVG(metric_value) AS metric_value,
               COUNT(*) AS obs_count
        FROM context_metric_observations
        WHERE metric_id IN ({ph_metrics})
          AND ticker IS NOT NULL
          AND ticker IN ({ph_tickers})
        GROUP BY ticker, DATE(event_time), metric_id
    """
    df = pd.read_sql_query(sql, con, params=[*CATALYST_METRICS.keys(), *sorted(tickers)])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "trade_date"])

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    value_wide = (
        df.pivot_table(index=["ticker", "trade_date"], columns="metric_id", values="metric_value", aggfunc="mean")
        .rename(columns=CATALYST_METRICS)
        .reset_index()
    )
    count_wide = (
        df.groupby(["ticker", "trade_date"], observed=True)["obs_count"]
        .sum()
        .reset_index()
        .rename(columns={"obs_count": "cat_doc_count"})
    )
    return value_wide.merge(count_wide, on=["ticker", "trade_date"], how="left")


def _add_sentiment_placeholders(panel: pd.DataFrame) -> pd.DataFrame:
    for col in SENTIMENT_PLACEHOLDER_COLS:
        if col not in panel.columns:
            panel[col] = np.nan
    panel["sentiment_available"] = 0
    return panel


def _availability_flags(panel: pd.DataFrame) -> pd.DataFrame:
    mmo_cols = [c for c in panel.columns if c.startswith("mmo_")]
    intra_cols = [c for c in panel.columns if c.startswith("intra_")]
    macro_cols = [c for c in panel.columns if c.startswith("macro_")]
    snap_cols = [c for c in panel.columns if c.startswith("snap_")]
    cat_cols = [c for c in CATALYST_METRICS.values() if c in panel.columns]

    panel["has_daily_metrics"] = panel[mmo_cols].notna().any(axis=1).astype(np.int8) if mmo_cols else 0
    panel["has_intraday_metrics"] = panel[intra_cols].notna().any(axis=1).astype(np.int8) if intra_cols else 0
    panel["has_macro"] = panel[macro_cols].notna().any(axis=1).astype(np.int8) if macro_cols else 0
    panel["has_snapshot"] = panel[snap_cols].notna().any(axis=1).astype(np.int8) if snap_cols else 0

    if cat_cols and panel["sentiment_available"].max() == 0:
        has_cat = panel[cat_cols].notna().any(axis=1)
        panel.loc[has_cat, "sentiment_available"] = 1

    return panel


def enrich_panel(panel: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    tickers = set(panel["ticker"].unique())
    con = sqlite3.connect(str(db_path))
    try:
        print("Enriching panel…")
        t0 = time.perf_counter()

        daily = _load_daily_metrics_bulk(con, tickers)
        print(f"  daily metrics: {len(daily):,} keys ({time.perf_counter() - t0:.1f}s)")

        t1 = time.perf_counter()
        intra = _load_intraday_daily_bulk(con, tickers)
        print(f"  intraday aggregates: {len(intra):,} keys ({time.perf_counter() - t1:.1f}s)")

        t2 = time.perf_counter()
        macro = _macro_by_date(con, panel["trade_date"])
        print(f"  macro dates: {len(macro):,} ({time.perf_counter() - t2:.1f}s)")

        t3 = time.perf_counter()
        snaps = _load_snapshots(con, tickers)
        print(f"  snapshots: {len(snaps):,} rows ({time.perf_counter() - t3:.1f}s)")

        t4 = time.perf_counter()
        catalyst = _load_catalyst_daily(con, tickers)
        print(f"  catalyst: {len(catalyst):,} keys ({time.perf_counter() - t4:.1f}s)")
    finally:
        con.close()

    out = panel.merge(daily, on=["ticker", "trade_date"], how="left")
    if not intra.empty:
        out = out.merge(intra, on=["ticker", "trade_date"], how="left")
    out = out.merge(macro, on="trade_date", how="left")
    if not snaps.empty:
        out = out.merge(snaps, on=["ticker", "trade_date"], how="left")
    if not catalyst.empty:
        out = out.merge(catalyst, on=["ticker", "trade_date"], how="left")
        cat_cols = [c for c in CATALYST_METRICS.values() if c in out.columns]
        out["sentiment_available"] = (
            out[cat_cols].notna().any(axis=1).astype(np.int8) if cat_cols else 0
        )
    else:
        out = _add_sentiment_placeholders(out)

    for col in ("news_tone", "news_volume"):
        if col not in out.columns:
            out[col] = np.nan

    return _availability_flags(out)


def column_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "base": [],
        "daily_metrics": [],
        "intraday": [],
        "macro": [],
        "snapshots": [],
        "sentiment": [],
        "availability": [],
    }
    for col in columns:
        if col in OUTPUT_COLUMNS:
            groups["base"].append(col)
        elif col.startswith("mmo_"):
            groups["daily_metrics"].append(col)
        elif col.startswith("intra_"):
            groups["intraday"].append(col)
        elif col.startswith("macro_"):
            groups["macro"].append(col)
        elif col.startswith("snap_"):
            groups["snapshots"].append(col)
        elif col in SENTIMENT_PLACEHOLDER_COLS or col == "sentiment_available":
            groups["sentiment"].append(col)
        elif col.startswith("has_"):
            groups["availability"].append(col)
        else:
            groups.setdefault("other", []).append(col)
    return {k: v for k, v in groups.items() if v}


def write_manifest(path: Path, panel: pd.DataFrame, *, db_path: Path, n_tickers_requested: int) -> None:
    groups = column_groups(panel.columns.tolist())
    manifest = {
        "version": 1,
        "source_db": str(db_path),
        "rows": len(panel),
        "columns": len(panel.columns),
        "tickers": int(panel["ticker"].nunique()),
        "tickers_requested": n_tickers_requested,
        "date_min": panel["trade_date"].min(),
        "date_max": panel["trade_date"].max(),
        "target_column": "fwd_ret_5d",
        "column_groups": groups,
        "availability_rates": {
            col: float(panel[col].mean())
            for col in panel.columns
            if col.startswith("has_") or col == "sentiment_available"
        },
        "non_null_rates": {
            group: float(panel[cols].notna().any(axis=1).mean())
            for group, cols in groups.items()
            if group not in ("base", "availability") and cols
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_full_panel(
    db_path: Path,
    *,
    n_tickers: int,
    fwd_days: int = 5,
) -> pd.DataFrame:
    base = build_base_panel(db_path, n_tickers=n_tickers, fwd_days=fwd_days)
    return enrich_panel(base, db_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full MIRA stock feature panel (Parquet)")
    parser.add_argument("--tickers", type=int, default=1000, help="Top N by liquidity")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite daily-bars DB path (or set MIRA_MARKET_DB)")
    parser.add_argument("--fwd", type=int, default=5, help="Forward return horizon (days)")
    parser.add_argument("--out", type=Path, default=None, help="Output Parquet path")
    parser.add_argument("--csv", action="store_true", help="Also write CSV alongside Parquet")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: market.db not found: {args.db}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    panel = build_full_panel(args.db, n_tickers=args.tickers, fwd_days=args.fwd)
    elapsed = time.perf_counter() - t0

    stem = f"stock_daily_features_full_{args.tickers}"
    parquet_path = args.out or (OUTPUT_DIR / f"{stem}.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(parquet_path, index=False)

    manifest_path = parquet_path.with_suffix(".manifest.json")
    write_manifest(manifest_path, panel, db_path=args.db, n_tickers_requested=args.tickers)

    print(f"\nWrote {parquet_path}")
    print(f"  manifest: {manifest_path}")
    print(f"  rows: {len(panel):,}")
    print(f"  columns: {len(panel.columns)}")
    print(f"  tickers: {panel['ticker'].nunique()}")
    print(f"  dates: {panel['trade_date'].min()} → {panel['trade_date'].max()}")
    print(f"  elapsed: {elapsed:.1f}s")

    if args.csv:
        csv_path = parquet_path.with_suffix(".csv")
        panel.to_csv(csv_path, index=False)
        csv_mb = csv_path.stat().st_size / (1024 * 1024)
        pq_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"  csv: {csv_path} ({csv_mb:.1f} MB)")
        print(f"  parquet size: {pq_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())