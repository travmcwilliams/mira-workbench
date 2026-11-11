"""Build MIRA stock_daily_features panel from a SQLite daily-bars market.db.

Produces a stacked panel CSV: one row per (trade_date, ticker) with technical
features and a forward-return target. Uses only point-in-time-safe inputs.

Usage:
    python scripts/build_stock_panel.py --tickers 500
    python scripts/build_stock_panel.py --tickers 1000 --db path/to/market.db
"""

from __future__ import annotations

import os
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.environ.get("MIRA_MARKET_DB", "market.db"))
OUTPUT_DIR = ROOT / "data" / "sample"

# Match existing MIRA sample schema
OUTPUT_COLUMNS = [
    "trade_date",
    "ticker",
    "sector",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "close_vs_sma20",
    "sma20_vs_sma50",
    "rsi_14",
    "atr_pct",
    "vol_20d_ann",
    "volume_z20",
    "hl_spread_bps",
    "dist_52w_high",
    "dist_52w_low",
    "gap_open",
    "fwd_ret_5d",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr / close.replace(0, np.nan)


def compute_ticker_features(bars: pd.DataFrame, *, fwd_days: int = 5) -> pd.DataFrame:
    """Build feature rows for one ticker from OHLCV daily bars."""
    df = bars.sort_values("trade_date").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    df["ret_1d"] = close.pct_change()
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    df["close_vs_sma20"] = close / sma20 - 1.0
    df["sma20_vs_sma50"] = sma20 / sma50 - 1.0
    df["rsi_14"] = _rsi(close, 14)
    df["atr_pct"] = _atr_pct(high, low, close, 14)

    ret1 = df["ret_1d"]
    df["vol_20d_ann"] = ret1.rolling(20, min_periods=20).std() * np.sqrt(252)

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std().replace(0, np.nan)
    df["volume_z20"] = (volume - vol_mean) / vol_std

    df["hl_spread_bps"] = ((high - low) / close.replace(0, np.nan)) * 10_000

    high_52w = high.rolling(252, min_periods=60).max()
    low_52w = low.rolling(252, min_periods=60).min()
    df["dist_52w_high"] = close / high_52w - 1.0
    df["dist_52w_low"] = close / low_52w - 1.0

    df["gap_open"] = open_ / close.shift(1) - 1.0
    df["fwd_ret_5d"] = close.shift(-fwd_days) / close - 1.0

    return df


def load_top_tickers(con: sqlite3.Connection, n: int) -> list[tuple[str, str, int]]:
    """Return top N tickers by avg dollar volume (liquidity proxy), excluding synthetics."""
    query = """
        SELECT s.symbol_id, s.ticker, COALESCE(s.sector, 'Unknown') AS sector,
               AVG(b.close * b.volume) AS avg_dollar_vol
        FROM symbols s
        JOIN universe_memberships um ON s.symbol_id = um.symbol_id
        JOIN bars_daily b ON b.symbol_id = s.symbol_id
        WHERE s.ticker NOT LIKE 'SYNTH_%'
          AND um.removed_at IS NULL
        GROUP BY s.symbol_id, s.ticker, s.sector
        HAVING COUNT(*) >= 200
        ORDER BY avg_dollar_vol DESC
        LIMIT ?
    """
    rows = con.execute(query, (n,)).fetchall()

    return [(r[1], r[2], r[0]) for r in rows]


def load_bars(con: sqlite3.Connection, symbol_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT trade_date, open, high, low, close, volume
        FROM bars_daily
        WHERE symbol_id = ?
        ORDER BY trade_date
        """,
        con,
        params=[symbol_id],
    )


def build_panel(
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

        print(f"Building panel for {len(tickers)} tickers from {db_path}")
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
                print(f"  processed {i}/{len(tickers)} tickers, rows so far: {sum(len(f) for f in frames):,}")

        if not frames:
            raise SystemExit("No feature rows produced.")

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
        return panel
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIRA stock feature panel CSV")
    parser.add_argument("--tickers", type=int, default=500, help="Top N by liquidity (500 or 1000)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite daily-bars DB path (or set MIRA_MARKET_DB)")
    parser.add_argument("--fwd", type=int, default=5, help="Forward return horizon (days)")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: market.db not found: {args.db}", file=sys.stderr)
        print("Set --db (or MIRA_MARKET_DB) to a SQLite daily-bars DB", file=sys.stderr)
        return 1

    panel = build_panel(args.db, n_tickers=args.tickers, fwd_days=args.fwd)
    out = args.out or OUTPUT_DIR / f"stock_daily_features_{args.tickers}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"  rows: {len(panel):,}")
    print(f"  tickers: {panel['ticker'].nunique()}")
    print(f"  dates: {panel['trade_date'].min()} → {panel['trade_date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())