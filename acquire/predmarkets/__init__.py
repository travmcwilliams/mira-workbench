"""Prediction-market acquisition (Polymarket + Kalshi) — API-first, no scraping.

5-year backfill of event-probability histories feeding a macro theme layer for
downstream consumers (ctx_pm_* daily metrics). Unlike the news campaign this is an ingest job:
both platforms expose public historical APIs, so Wayback is not involved.

Modules: probe (census / discovery), then store / polymarket / kalshi / campaign.
"""
