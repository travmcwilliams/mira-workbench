"""
agents/rules_monitor.py
Policy / Rules Monitor (Agent 9, LLM-driven).

On a schedule, fetch each source's rules/policies page(s) and use LLM to detect material changes.
Alerts when e.g. eligibility, scoring formula, deadlines, or filing requirements change.

Requires LLM-API-KEY.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.db import get_db
from utils.secrets import get_llm_api_key


def scan_for_policy_changes():
    key = get_llm_api_key()
    if not key:
        print("[rules_monitor] No LLM-API-KEY configured. Skipping LLM calls.")

    print("[rules_monitor] Scanning source policy pages for changes (LLM)...")
    get_db()
    # TODO: for each source in registry that has a policy URL:
    #   fetch page (playwright if JS heavy), send excerpt + previous known to LLM,
    #   if delta: write alert + update last_known_rules snapshot.
    print("[rules_monitor] (stub) Implement LLM change detection + alert emission.")
    return 0
