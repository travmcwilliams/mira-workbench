"""Generate the synthetic restaurant sample datasets in ``data/sample/``.

Both files are fabricated from a fixed seed. They mirror the *shape* of a typical
POS labor export (one row per employee x role) and a daily sales/labor join, so the
restaurant feature engineering, dashboards and tests have something realistic to run
on. No real business, employee, or transaction is represented.

Usage:
    python scripts/make_synthetic_restaurant_samples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
SEED = 20260101

FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
         "Quinn", "Drew", "Parker", "Reese", "Rowan", "Skyler", "Emerson", "Hayden",
         "Kendall", "Logan", "Peyton", "Sage", "Blake", "Cameron", "Dakota", "Elliot"]
LAST = ["Abbott", "Barnes", "Chen", "Delgado", "Ellis", "Foster", "Garcia", "Hughes",
        "Ibarra", "Jensen", "Kim", "Lopez", "Moreno", "Nguyen", "Ortiz", "Patel",
        "Quint", "Reyes", "Singh", "Turner", "Underwood", "Vance", "Walsh", "Young",
        "Archer", "Bishop", "Carver", "Dalton", "Easton", "Fleming", "Gallo", "Hartley",
        "Irwin", "Jacobs", "Keller", "Larkin", "Mercer", "Nash", "Oakley", "Pruitt",
        "Quinlan", "Ramsey", "Sutton", "Thorne", "Upton", "Vargas", "Whitaker", "Yates",
        "Zamora", "Ashby", "Brennan", "Cortez", "Dunn", "Esposito", "Farrell", "Grant",
        "Holloway", "Ingram", "Joyce", "Kearney", "Lyons", "Mahoney", "Norris", "Osborne"]

# role -> (headcount, base hourly wage, tipped, hours range)
ROLES = {
    "Server": (12, 5.62, True, (120, 700)),
    "Bartender": (7, 5.62, True, (100, 650)),
    "Host": (7, 13.0, False, (40, 350)),
    "Busser/Runner": (15, 9.0, True, (30, 400)),
    "Dish/Porter": (4, 17.0, False, (200, 650)),
    "Manager": (5, 24.0, False, (300, 900)),
    "Training": (9, 13.0, False, (5, 40)),
}

LABOR_COLS = ["Employee ID", "Last Name", "First Name", "Role", "Reg Hrs", "1.5x Hrs",
              "PayPerHour", "Total Pay", "Net Sales", "Non Revenue Receipts", "Auto Grat",
              "Cash Tips", "Credit Tips", "Tender Tips", "Total Tips", "Declared Tips",
              "Contributions", "Distributions"]


def make_labor(rng: np.random.Generator) -> pd.DataFrame:
    # One distinct surname per employee, as in a real roster.
    lasts = list(rng.permutation(LAST))
    names = [(str(l), str(rng.choice(FIRST))) for l in lasts]
    rows: list[dict] = []
    i = 0
    for role, (n, wage, tipped, (lo, hi)) in ROLES.items():
        for _ in range(n):
            last, first = names[i]
            i += 1
            rows.append(_labor_row(rng, last, first, role, wage, tipped, lo, hi))
    # One deliberately multi-role employee so role-mix features have a case to find.
    for role in ("Server", "Bartender", "Host"):
        n, wage, tipped, (lo, hi) = ROLES[role]
        rows.append(_labor_row(rng, "Rivera", "Sam", role, wage, tipped, lo // 2, hi // 3))
    return pd.DataFrame(rows, columns=LABOR_COLS)


def _labor_row(rng, last, first, role, wage, tipped, lo, hi) -> dict:
    reg = round(float(rng.uniform(lo, hi)), 2)
    ot = round(float(max(0.0, rng.normal(reg * 0.04, reg * 0.05))), 2)
    pay = round(wage * reg + wage * 1.5 * ot, 2)
    sales = round(reg * float(rng.uniform(110, 210)), 2) if role in {"Server", "Bartender"} else 0.0
    credit = round(sales * float(rng.uniform(0.15, 0.21)), 2) if tipped and sales else (
        round(reg * float(rng.uniform(2, 6)), 2) if tipped else 0.0)
    cash = round(credit * float(rng.uniform(0.0, 0.05)), 2)
    tender = round(float(rng.uniform(0, 60)), 2) if tipped else 0.0
    total = round(credit + cash + tender, 2)
    return {
        "Employee ID": "", "Last Name": last, "First Name": first, "Role": role,
        "Reg Hrs": reg, "1.5x Hrs": ot, "PayPerHour": wage, "Total Pay": pay,
        "Net Sales": sales, "Non Revenue Receipts": round(float(rng.choice([0, 0, 0, 50, 150])), 2),
        "Auto Grat": 0.0, "Cash Tips": cash, "Credit Tips": credit, "Tender Tips": tender,
        "Total Tips": total, "Declared Tips": 0.0, "Contributions": credit,
        "Distributions": round(total * float(rng.uniform(0.9, 1.1)), 2),
    }


def make_daily(rng: np.random.Generator, days: int = 860) -> pd.DataFrame:
    dates = pd.date_range("2023-11-01", periods=days, freq="D")
    dow_mult = {0: 0.55, 1: 0.6, 2: 0.7, 3: 0.85, 4: 1.3, 5: 1.45, 6: 1.0}
    rows = []
    for d in dates:
        season = 1 + 0.12 * np.sin(2 * np.pi * (d.dayofyear - 120) / 365)
        trend = 1 + 0.15 * (d - dates[0]).days / days
        net = max(0.0, float(rng.normal(6500, 900)) * dow_mult[d.dayofweek] * season * trend)
        comps = round(net * float(rng.uniform(0.005, 0.03)), 2)
        guests = int(max(0, net / float(rng.uniform(38, 52))))
        checks = int(max(0, guests / float(rng.uniform(1.8, 2.6))))
        hours = max(0.0, 60 + net / float(rng.uniform(85, 120)) + float(rng.normal(0, 6)))
        ot = max(0.0, float(rng.normal(hours * 0.03, 2)))
        wages = hours * float(rng.uniform(10.5, 13.5))
        tips = net * float(rng.uniform(0.16, 0.2))
        tax = net * 0.1
        rows.append({
            "trading_day_date": d.date().isoformat(),
            "day_of_week": d.day_name(),
            "month": d.strftime("%Y-%m"),
            "week_start": (d - pd.Timedelta(days=d.dayofweek)).date().isoformat(),
            "net_sales": round(net, 2), "tax": round(tax, 2), "gross": round(net + tax, 2),
            "comps": comps, "guests": guests, "checks": checks,
            "avg_check": round(net / checks, 2) if checks else 0.0,
            "labor_hours": round(hours, 2), "regular_hours": round(hours - ot, 2),
            "ot_hours": round(ot, 2), "wages": round(wages, 2), "tips": round(tips, 2),
            "labor_pct": round(wages / net, 4) if net else 0.0,
            "sales_per_labor_hour": round(net / hours, 2) if hours else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    make_labor(rng).to_csv(OUT / "restaurant_labor_sample.csv", index=False)
    make_daily(rng).to_csv(OUT / "restaurant_daily_sample.csv", index=False)
    print(f"wrote {OUT / 'restaurant_labor_sample.csv'} and {OUT / 'restaurant_daily_sample.csv'}")


if __name__ == "__main__":
    main()
