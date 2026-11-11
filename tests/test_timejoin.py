"""Offline edge-case tests for core/timejoin.py — weekends, pre-listing, gap cap.

These pin the semantics the module was promoted FOR: the flaws of the original
``acquire/news/market.py::map_to_next_trading_day`` (phantom pre-listing
mappings, unbounded gaps) must stay fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.timejoin import (
    asof_join,
    attach_next_trading_day,
    collapse_to_trading_day,
    forward_returns,
    next_trading_day_map,
)

# Mon-Fri week, its Monday follow-up, then a 15-calendar-day hole (e.g. a halt).
TD = ["2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18",
      "2021-06-21", "2021-07-06"]


# --- next_trading_day_map -----------------------------------------------------

def test_weekend_rolls_forward_to_monday():
    m = next_trading_day_map(["2021-06-19", "2021-06-20"], TD)
    assert m == {"2021-06-19": "2021-06-21", "2021-06-20": "2021-06-21"}


def test_trading_day_maps_to_itself():
    assert next_trading_day_map(["2021-06-15"], TD) == {"2021-06-15": "2021-06-15"}


def test_pre_listing_dropped_by_default():
    # 2021-06-10 precedes the first session: mapping it to 06-14 would be a phantom.
    assert next_trading_day_map(["2021-06-10"], TD) == {}


def test_pre_listing_opt_in_still_gap_capped():
    m = next_trading_day_map(["2021-06-10", "2021-05-01"], TD, include_pre_listing=True)
    assert m == {"2021-06-10": "2021-06-14"}   # 4-day gap ok; 44-day gap refused


def test_gap_cap_refuses_long_holes():
    # 2021-06-24's next session is 2021-07-06 — 12 calendar days away.
    assert next_trading_day_map(["2021-06-24"], TD) == {}
    assert next_trading_day_map(["2021-06-24"], TD, max_gap_days=12) == \
        {"2021-06-24": "2021-07-06"}
    assert next_trading_day_map(["2021-06-24"], TD, max_gap_days=None) == \
        {"2021-06-24": "2021-07-06"}


def test_after_last_session_dropped():
    assert next_trading_day_map(["2021-07-07"], TD) == {}


def test_empty_trading_days():
    assert next_trading_day_map(["2021-06-15"], []) == {}


# --- attach_next_trading_day ----------------------------------------------------

def test_attach_drops_unmappable_and_preserves_order():
    df = pd.DataFrame({"date": ["2021-06-19", "2021-06-10", "2021-06-15"],
                       "x": [1, 2, 3]})
    out = attach_next_trading_day(df, TD)
    assert out["date"].tolist() == ["2021-06-19", "2021-06-15"]
    assert out["trading_date"].tolist() == ["2021-06-21", "2021-06-15"]
    assert out["x"].tolist() == [1, 3]


# --- collapse_to_trading_day (weekend-merge dedup) -------------------------------

def test_collapse_keeps_freshest_calendar_date():
    df = pd.DataFrame({
        "date": ["2021-06-18", "2021-06-19", "2021-06-20"],
        "trading_date": ["2021-06-21", "2021-06-21", "2021-06-21"],
        "val": [1.0, 2.0, 3.0],
    })
    out = collapse_to_trading_day(df)
    assert len(out) == 1
    assert out.loc[0, "date"] == "2021-06-20"     # latest info still <= the session
    assert out.loc[0, "val"] == 3.0


def test_collapse_respects_group_key():
    df = pd.DataFrame({
        "event_type": ["cpi", "cpi", "nfp"],
        "date": ["2021-06-19", "2021-06-20", "2021-06-19"],
        "trading_date": ["2021-06-21"] * 3,
        "val": [1.0, 2.0, 9.0],
    })
    out = collapse_to_trading_day(df, by="event_type")
    assert len(out) == 2
    assert out.set_index("event_type").loc["cpi", "val"] == 2.0
    assert out.set_index("event_type").loc["nfp", "val"] == 9.0


def test_collapse_agg_path():
    df = pd.DataFrame({
        "date": ["2021-06-19", "2021-06-20", "2021-06-15"],
        "trading_date": ["2021-06-21", "2021-06-21", "2021-06-15"],
        "val": [1.0, 2.0, 5.0],
    })
    out = collapse_to_trading_day(df, agg={"val": "sum"})
    got = dict(zip(out["trading_date"], out["val"], strict=True))
    assert got == {"2021-06-15": 5.0, "2021-06-21": 3.0}


# --- asof_join -------------------------------------------------------------------

def test_asof_backward_takes_latest_at_or_before():
    left = pd.DataFrame({"date": ["2021-06-16", "2021-06-21"], "ev": ["a", "b"]})
    right = pd.DataFrame({"date": ["2021-06-14", "2021-06-18"], "val": [1.0, 2.0]})
    out = asof_join(left, right)
    assert out["val"].tolist() == [1.0, 2.0]
    assert out["date_right"].tolist() == ["2021-06-14", "2021-06-18"]


def test_asof_tolerance_refuses_stale():
    left = pd.DataFrame({"date": ["2021-06-16", "2021-06-21"]})
    right = pd.DataFrame({"date": ["2021-06-14", "2021-06-18"], "val": [1.0, 2.0]})
    out = asof_join(left, right, tolerance_days=2)
    assert out["val"].tolist()[0] == 1.0          # 2 days back: allowed
    assert np.isnan(out["val"].tolist()[1])       # 3 days back: stale


def test_asof_lookahead_requires_explicit_opt_in():
    left = pd.DataFrame({"date": ["2021-06-16"]})
    right = pd.DataFrame({"date": ["2021-06-18"], "val": [2.0]})
    with pytest.raises(ValueError, match="lookahead"):
        asof_join(left, right, direction="forward")
    out = asof_join(left, right, direction="forward", allow_lookahead=True)
    assert out["val"].tolist() == [2.0]


# --- forward_returns -------------------------------------------------------------

def test_forward_returns_strictly_future():
    closes = pd.DataFrame({"date": ["2021-06-14", "2021-06-15", "2021-06-16"],
                           "close": [1.0, 2.0, 4.0]})
    out = forward_returns(closes, horizons=(1,))
    assert out["fwd_ret_1d"].tolist()[:2] == [1.0, 1.0]
    assert np.isnan(out["fwd_ret_1d"].iloc[-1])
