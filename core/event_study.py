"""Generalized event study: ANY events table -> forward-return medians vs baseline.

The macro-release analogue of the news studies: condition forward returns on a
table of dated, typed events instead of a continuous sentiment series. Medians,
not means — the replicated news findings showed the means are outlier-skewed
(see ``acquire/news/README.md``); every estimate ships with a circular
block-bootstrap confidence interval so sparse event samples and overlapping
multi-day horizons don't manufacture false precision.

    events:  DataFrame with at least ``date`` (ISO) + a group column
             (default ``event_type``) — e.g. ``events.store.fetch_events(...)``
    closes:  DataFrame with ``date`` + ``close`` (trading days only) — e.g.
             ``acquire/news/market.py::fetch_daily_close`` or an offline CSV

Date semantics come from ``core.timejoin`` (the promoted leak-safe logic):
weekend/holiday events roll FORWARD to the next session, pre-listing events are
dropped (no phantom mappings), gaps are capped, and multiple event rows landing
on one session are collapsed per event type (weekend-merge dedup). A release on
trading day *t* is paired with returns from *t*'s close onward — for an 08:30
print the day-*t* close already reflects the announcement, so what's measured
is post-event drift, not the announcement jump.

There was no pre-existing block-bootstrap machinery in the repo; this module is
its home (``block_bootstrap_ci``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from core.timejoin import (
    DEFAULT_MAX_GAP_DAYS,
    attach_next_trading_day,
    collapse_to_trading_day,
    forward_returns,
)


def block_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    block_len: int = 5,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the MEDIAN of a time-ordered sample.

    Resamples contiguous blocks (wrapping at the end) so serial dependence —
    overlapping multi-day horizons, volatility clustering — survives into the
    replicates instead of being bootstrapped away. Deterministic per ``seed``.
    NaNs are ignored; an empty sample yields ``(nan, nan)``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    b = max(1, min(block_len, n))
    k = -(-n // b)  # blocks per replicate (ceil)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    samples = arr[idx.reshape(n_boot, k * b)[:, :n]]
    stats = np.median(samples, axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def run_event_study(
    events: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 5),
    group_col: str = "event_type",
    date_col: str = "date",
    max_gap_days: int | None = DEFAULT_MAX_GAP_DAYS,
    n_boot: int = 2000,
    block_len: int = 5,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Forward-return medians on event sessions vs the unconditional baseline.

    Returns one tidy row per (event group, horizon):

        [group_col, horizon_days, n_events, median_fwd_ret, ci_lo, ci_hi,
         baseline_median, baseline_lo, baseline_hi, delta_median]

    ``baseline_*`` is the same statistic over ALL trading days in ``closes``
    (the null the event median is read against); ``delta_median`` is
    event-minus-baseline. Groups whose events all fail to map (pre-listing,
    over-gap, past end of data) simply produce no rows.
    """
    if group_col not in events.columns:
        raise KeyError(f"events table has no {group_col!r} column")
    if date_col not in events.columns:
        raise KeyError(f"events table has no {date_col!r} column")

    horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    rets = forward_returns(closes, horizons)

    ev = attach_next_trading_day(
        events, rets["date"].tolist(),
        date_col=date_col, max_gap_days=max_gap_days)
    ev = collapse_to_trading_day(ev, by=group_col, date_col=date_col)
    joined = ev.merge(
        rets.rename(columns={"date": "trading_date"}),
        on="trading_date", how="inner")

    rows: list[dict[str, object]] = []
    for h in horizons:
        col = f"fwd_ret_{h}d"
        base = rets[col].dropna().to_numpy()
        base_med = float(np.median(base)) if base.size else float("nan")
        base_lo, base_hi = block_bootstrap_ci(
            base, n_boot=n_boot, block_len=block_len, ci=ci, seed=seed)
        for gval, g in joined.groupby(group_col, sort=True):
            vals = g.sort_values("trading_date")[col].dropna().to_numpy()
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            lo, hi = block_bootstrap_ci(
                vals, n_boot=n_boot, block_len=block_len, ci=ci, seed=seed)
            rows.append({
                group_col: gval,
                "horizon_days": h,
                "n_events": int(vals.size),
                "median_fwd_ret": med,
                "ci_lo": lo,
                "ci_hi": hi,
                "baseline_median": base_med,
                "baseline_lo": base_lo,
                "baseline_hi": base_hi,
                "delta_median": med - base_med,
            })
    return pd.DataFrame(rows)
