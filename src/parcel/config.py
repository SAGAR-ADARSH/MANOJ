from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "config" / "policy.json"


def load_policy(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_POLICY
    with path.open() as f:
        policy = json.load(f)
    required = {
        "daily_capacity", "per_carrier_capacity", "missed_late_cost",
        "intervention_success_probability", "prediction_local_hour", "timezone",
        "max_promised_days", "currency",
    }
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"Policy missing keys: {sorted(missing)}")
    if not 0 < policy["daily_capacity"] <= 100_000:
        raise ValueError("daily_capacity must be in (0, 100000]")
    if not 0 < policy["per_carrier_capacity"] <= policy["daily_capacity"]:
        raise ValueError("per_carrier_capacity must be positive and <= daily_capacity")
    if policy["missed_late_cost"] <= 0:
        raise ValueError("missed_late_cost must be positive")
    if not 0 < policy["intervention_success_probability"] <= 1:
        raise ValueError("intervention_success_probability must be in (0, 1]")
    if policy["timezone"] != "Asia/Kolkata" or policy["prediction_local_hour"] != 6:
        raise ValueError("This prototype is defined for a 06:00 Asia/Kolkata decision snapshot")
    return policy
