"""Tests for canonical records dataset profile."""

from __future__ import annotations

import pandas as pd

from core.dataset_profiles import detect_dataset_profile, profile_label


def test_detect_canonical_records_profile():
    df = pd.DataFrame({
        "record_id": ["1", "2"],
        "entity_name": ["A", "B"],
        "location_text": ["X", "Y"],
        "primary_value": [10.0, 20.0],
        "period": ["2024-01", "2024-02"],
        "source_name": ["demo", "demo"],
        "source_id": ["REAL001", "REAL001"],
        "region": ["00", "00"],
    })
    assert detect_dataset_profile(df) == "canonical_records"
    assert "MIRA" in profile_label("canonical_records")