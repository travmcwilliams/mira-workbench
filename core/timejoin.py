"""Leak-safe calendar-date <-> trading-day joins — shared, tested infrastructure.

Promoted from the hand-rolled logic in the news studies (``acquire/news/market.py``
``map_to_next_trading_day`` + the per-script merge/aggregate blocks in
``scripts/news_*_study.py``), with the two known flaws of the original fixed:

  * **pre-listing phantom mappings** — a date before a series' first trading day used
    to map onto its first session (2019 "news" landing on a 2021 IPO's first close).
    Such dates are dropped by default now (``include_pre_listing=True`` restores the
    old behavior, still subject to the gap cap).
  * **unbounded gaps** — a date could map across an arbitrarily long data hole
    (delisting window, source outage) and pair an event with a session weeks away.
    ``max_gap_days`` caps the calendar distance (default 7: spans any weekend plus a
    holiday cluster, refuses multi-week holes; ``None`` disables the cap).

Direction is always FORWARD to the next session (weekend/holiday events roll to
Monday, never back — no lookahead), and multiple calendar dates landing on one
session are collapsed explicitly (``collapse_to_trading_day``) so a Monday never
double-counts its weekend. Feature-style joins go the other way: ``asof_join`` pairs
each left row with the latest right row at-or-before it ("known as of", backward
only unless the caller explicitly opts into lookahead).

All dates are ISO ``YYYY-MM-DD`` strings (the repo-wide convention), which sort
lexicographically in chronological order.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date as _date
from typing import Iterable, Mapping

import pandas as pd

DEFAULT_MAX_GAP_DAYS = 7


def next_trading_day_map(
    dates: Iterable[str],
    trading_days: Iterable[str],
    *,
    max_gap_days: int | None = DEFAULT_MAX_GAP_DAYS,
    include_pre_listing: bool = False,
) -> dict[str, str]:
    """Map each calendar date to the next trading day >= it (weekend -> Monday).

    A date is left out of the result (no phantom pairing) when:
      * it falls after the last trading day, or
      * it precedes the first trading day (pre-listing) and ``include_pre_listing``
        is False, or
      * the mapped session is more than ``max_gap_days`` calendar days away.
    """
    td = sorted(set(trading_days))
    out: dict[str, str] = {}
    if not td:
        return out
    first = td[0]
    for d in dates:
        if not isinstance(d, str):   # NaN / None in a date column: unmappable, drop
            continue
        if d < first and not include_pre_listing:
            continue
        idx = bisect_left(td, d)
        if idx >= len(td):
            continue
        mapped = td[idx]
        if max_gap_days is not None:
            gap = (_date.fromisoformat(mapped) - _date.fromisoformat(d)).days
            if gap > max_gap_days:
                continue
        out[d] = mapped
    return out


def attach_next_trading_day(
    df: pd.DataFrame,
    trading_days: Iterable[str],
    *,
    date_col: str = "date",
    out_col: str = "trading_date",
    max_gap_days: int | None = DEFAULT_MAX_GAP_DAYS,
    include_pre_listing: bool = False,
) -> pd.DataFrame:
    """Add ``out_col`` = next trading day for ``date_col``; drop unmappable rows.

    Row order is preserved for the rows that survive. Dropping (rather than NaN)
    is deliberate: an unmappable event must never silently join a return.
    """
    mapping = next_trading_day_map(
        df[date_col].tolist(), trading_days,
        max_gap_days=max_gap_days, include_pre_listing=include_pre_listing)
    out = df[df[date_col].isin(mapping)].copy()
    out[out_col] = out[date_col].map(mapping)
    return out.reset_index(drop=True)


def collapse_to_trading_day(
    df: pd.DataFrame,
    *,
    on: str = "trading_date",
    date_col: str = "date",
    by: str | None = None,
    agg: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Collapse multiple calendar dates that share one session (weekend-merge dedup).

    Without ``agg``, keeps the LAST calendar date's row per session (the freshest
    information that is still <= the session — leak-safe). With ``agg`` (a
    column -> pandas-aggregation mapping), aggregates all rows per session instead.
    ``by`` adds a grouping key (e.g. ``event_type``) so distinct event streams are
    collapsed independently.
    """
    keys = [by, on] if by else [on]
    if agg is not None:
        out = df.groupby(keys, as_index=False).agg(dict(agg))
        return out.sort_values(keys).reset_index(drop=True)
    out = df.sort_values(keys + [date_col], kind="stable")
    out = out.drop_duplicates(subset=keys, keep="last")
    return out.reset_index(drop=True)


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str = "date",
    right_on: str | None = None,
    by: str | None = None,
    tolerance_days: int | None = None,
    direction: str = "backward",
    allow_lookahead: bool = False,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """As-of join: pair each left row with the latest right row at-or-before its date.

    The leak-safe direction is ``backward`` ("what was known as of this date"); any
    other direction reads the future and requires ``allow_lookahead=True`` so a
    study can never do it by accident. ``tolerance_days`` refuses stale matches
    (right row older than the tolerance -> right columns are NaN).

    Both key columns are parsed with ``pd.to_datetime``; the result is sorted by
    the left key (a ``merge_asof`` requirement) and the temp key is dropped.
    """
    if direction != "backward" and not allow_lookahead:
        raise ValueError(
            f"asof_join(direction={direction!r}) reads the future; "
            "pass allow_lookahead=True if that is genuinely intended")
    rk = right_on or on
    lf = left.copy()
    rf = right.copy()
    lf["_asof_ts"] = pd.to_datetime(lf[on])
    rf["_asof_ts"] = pd.to_datetime(rf[rk])
    lf = lf.sort_values("_asof_ts", kind="stable")
    rf = rf.sort_values("_asof_ts", kind="stable")
    tol = pd.Timedelta(days=tolerance_days) if tolerance_days is not None else None
    merged = pd.merge_asof(
        lf, rf, on="_asof_ts", by=by, tolerance=tol,
        direction=direction, suffixes=suffixes)
    return merged.drop(columns=["_asof_ts"]).reset_index(drop=True)


def forward_returns(
    closes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
) -> pd.DataFrame:
    """Add ``fwd_ret_{h}d`` columns = close[t+h]/close[t] - 1 (leak-safe: strictly future).

    Same contract as ``acquire/news/market.py::forward_returns`` — duplicated here so
    core stays importable without the acquire tree; the news scripts migrate to this
    copy as a follow-up.
    """
    out = closes.sort_values("date").reset_index(drop=True).copy()
    for h in horizons:
        out[f"fwd_ret_{h}d"] = out["close"].shift(-h) / out["close"] - 1.0
    return out
