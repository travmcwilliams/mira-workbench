"""CLI wiring for `mira macro` — offline (no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ACQUIRE = ROOT / "acquire"
if str(_ACQUIRE) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE))

import macro.ecb as ecb_mod  # noqa: E402
import macro.fred as fred_mod  # noqa: E402
from macro.store import open_db, upsert_rows  # noqa: E402

import mira_cli  # noqa: E402

FAKE_FRED_ROWS = [
    {
        "series_id": "DGS10",
        "observation_date": "2024-01-02",
        "value": 4.0,
        "release_timestamp": "2024-01-03T00:00:00Z",
        "available_for_signal_at": "2024-01-03T00:00:00Z",
        "vintage": "current",
        "source": "fred",
        "frequency": "daily",
    }
]


def test_macro_list_empty(tmp_path, capsys):
    db = tmp_path / "m.duckdb"
    rc = mira_cli.main(["macro", "list", "--db", str(db)])
    assert rc == 0
    assert "no macro series" in capsys.readouterr().out


def test_macro_backfill_fred_mocked(tmp_path, capsys):
    db = tmp_path / "m.duckdb"
    with patch.object(fred_mod, "fetch_fred_basket", return_value=FAKE_FRED_ROWS):
        rc = mira_cli.main(
            [
                "macro", "backfill",
                "--source", "fred",
                "--series", "DGS10",
                "--db", str(db),
            ]
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "upserted 1" in out

    rc = mira_cli.main(["macro", "list", "--db", str(db)])
    assert rc == 0
    assert "DGS10" in capsys.readouterr().out


def test_macro_backfill_ecb_mocked(tmp_path, capsys):
    db = tmp_path / "m.duckdb"
    fake = [
        {
            "series_id": "ECB_USD",
            "observation_date": "2024-01-02",
            "value": 1.1,
            "release_timestamp": "2024-01-02T15:00:00Z",
            "available_for_signal_at": "2024-01-02T15:00:00Z",
            "vintage": "2024-01-02",
            "source": "ecb",
            "frequency": "daily",
        }
    ]
    with patch.object(ecb_mod, "fetch_ecb_fx", return_value=fake):
        rc = mira_cli.main(
            [
                "macro", "backfill",
                "--source", "ecb",
                "--series", "USD",
                "--db", str(db),
            ]
        )
    assert rc == 0
    assert "upserted 1" in capsys.readouterr().out


def test_macro_asof_cli(tmp_path, capsys):
    db = tmp_path / "m.duckdb"
    con = open_db(db)
    upsert_rows(
        con,
        [
            {
                "series_id": "VIXCLS",
                "observation_date": "2024-06-01",
                "value": 14.2,
                "release_timestamp": "2024-06-02T00:00:00Z",
                "available_for_signal_at": "2024-06-02T00:00:00Z",
                "vintage": "current",
                "source": "fred",
                "frequency": "daily",
            }
        ],
    )
    con.close()

    rc = mira_cli.main(["macro", "asof", "--at", "2024-06-01", "--db", str(db)])
    assert rc == 0
    assert "nothing available" in capsys.readouterr().out

    rc = mira_cli.main(["macro", "asof", "--at", "2024-06-03", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VIXCLS" in out
    assert "14.2" in out
