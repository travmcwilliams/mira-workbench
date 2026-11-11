"""
registry/config.py
Load and validate the config-as-data pack (weights, caps, inclusion rules, seeds).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .scoring import validate_weights

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "source_registry.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read + validate the registry pack. Raises on malformed weights."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_weights(cfg.get("scoring_weights", {}))
    cfg.setdefault("norm_caps", {})
    cfg.setdefault("inclusion_rules", [])
    cfg.setdefault("categories", [])
    cfg.setdefault("platforms", [])
    cfg.setdefault("sources", [])
    return cfg


@lru_cache(maxsize=1)
def default_config() -> dict[str, Any]:
    return load_config()
