"""End-to-end synthetic event study + block-bootstrap unit tests. Fully offline.

The synthetic world: 120 business days, flat prices EXCEPT a +2% close-to-close
jump out of every "cpi" event session. The study must recover a +2% event-day
median against a 0% baseline — through the real DuckDB store, the real
timejoin mapping (weekend roll, pre-listing drop, weekend-merge dedup), and the
real bootstrap. A "noise" event type sits on flat sessions as the control.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.event_study import block_bootstrap_ci, run_event_study

_ACQUIRE = Path(__file__).resolve().parents[1] / "acquire"
if str(_ACQUIRE) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE))

from events import store  # noqa: E402

# --- block_bootstrap_ci --------------------------------------------------------

def test_bootstrap_constant_series_collapses():
    assert block_bootstrap_ci([0.5] * 20) == (0.5, 0.5)


def test_bootstrap_degenerate_inputs():
    lo, hi = block_bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)
    assert block_bootstrap_ci([1.0]) == (1.0, 1.0)


def test_bootstrap_deterministic_and_ordered():
    rng = np.random.default_rng(42)
    vals = rng.normal(0.0, 1.0, size=60)
    a = block_bootstrap_ci(vals, seed=7)
    b = block_bootstrap_ci(vals, seed=7)
    assert a == b
    lo, hi = a
    assert lo <= np.median(vals) <= hi


def test_bootstrap_ignores_nans():
    lo, hi = block_bootstrap_ci([0.5, float("nan"), 0.5, 0.5])
    assert (lo, hi) == (0.5, 0.5)


# --- synthetic end-to-end study --------------------------------------------------

# 120 business days from Mon 2021-01-04; index 10 == 2021-01-18 (a Monday here —
# this is OUR synthetic exchange calendar, not the NYSE's).
DAYS = pd.bdate_range("2021-01-04", periods=120).strftime("%Y-%m-%d").tolist()
EVENT_POS = [15, 25, 35, 45, 55, 65, 75, 85, 95]     # explicit cpi sessions
SATURDAY = "2021-01-16"                              # rolls forward to DAYS[10]
JUMP_SESSIONS = set(EVENT_POS) | {10}                # every cpi session jumps +2%
NOISE_POS = [20, 40, 60]                             # control events on flat days


def _closes() -> pd.DataFrame:
    close = [100.0]
    for t in range(len(DAYS) - 1):
        close.append(close[-1] * (1.02 if t in JUMP_SESSIONS else 1.0))
    return pd.DataFrame({"date": DAYS, "close": close})


def _event_rows() -> list[dict]:
    rows = [{"event_type": "cpi", "release_datetime": f"{DAYS[p]} 00:00:00",
             "source": "synthetic"} for p in EVENT_POS]
    # weekend release: must merge onto Monday DAYS[10], not be dropped
    rows.append({"event_type": "cpi", "release_datetime": f"{SATURDAY} 00:00:00",
                 "source": "synthetic"})
    # duplicate observation of DAYS[15] from a second source: same session ->
    # weekend-merge dedup must count it once
    rows.append({"event_type": "cpi", "release_datetime": f"{DAYS[15]} 00:00:00",
                 "source": "synthetic-alt"})
    # pre-listing phantom: before the first trading day -> must be dropped
    rows.append({"event_type": "cpi", "release_datetime": "2020-12-15 00:00:00",
                 "source": "synthetic"})
    rows += [{"event_type": "noise", "release_datetime": f"{DAYS[p]} 00:00:00",
              "source": "synthetic"} for p in NOISE_POS]
    return rows


def test_synthetic_event_study_end_to_end(tmp_path):
    con = store.open_db(tmp_path / "events.duckdb")
    try:
        # 9 explicit + weekend + alt-source dup + pre-listing + 3 noise = 15 rows
        assert store.ingest_events(con, _event_rows()) == 15
        events = store.fetch_events(con)
    finally:
        con.close()

    result = run_event_study(events, _closes(), horizons=(1, 5),
                             n_boot=500, seed=0)
    r = result.set_index(["event_type", "horizon_days"])

    # 9 explicit + 1 weekend-rolled; the duplicate deduped, pre-listing dropped
    cpi_1d = r.loc[("cpi", 1)]
    assert cpi_1d["n_events"] == 10
    assert cpi_1d["median_fwd_ret"] == pytest.approx(0.02, rel=1e-9)
    assert cpi_1d["baseline_median"] == pytest.approx(0.0, abs=1e-12)
    assert cpi_1d["delta_median"] == pytest.approx(0.02, rel=1e-9)
    assert cpi_1d["ci_lo"] <= cpi_1d["median_fwd_ret"] <= cpi_1d["ci_hi"]

    # each event's 5-day window contains exactly its own jump
    cpi_5d = r.loc[("cpi", 5)]
    assert cpi_5d["median_fwd_ret"] == pytest.approx(0.02, rel=1e-9)
    assert cpi_5d["baseline_median"] == pytest.approx(0.0, abs=1e-12)

    # control events on flat sessions: no effect
    noise_1d = r.loc[("noise", 1)]
    assert noise_1d["n_events"] == 3
    assert noise_1d["median_fwd_ret"] == pytest.approx(0.0, abs=1e-12)
    assert noise_1d["delta_median"] == pytest.approx(0.0, abs=1e-12)


def test_study_requires_columns():
    closes = _closes()
    with pytest.raises(KeyError, match="event_type"):
        run_event_study(pd.DataFrame({"date": [DAYS[0]]}), closes)
    with pytest.raises(KeyError, match="date"):
        run_event_study(pd.DataFrame({"event_type": ["cpi"]}), closes)


def test_study_unmappable_group_yields_no_rows():
    # every event pre-listing -> attach drops all -> empty result, no crash
    events = pd.DataFrame({"event_type": ["cpi"], "date": ["2020-01-01"]})
    assert run_event_study(events, _closes()).empty
