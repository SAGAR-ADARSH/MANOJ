from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_INPUT = [
    "shipment_id", "snapshot_at", "booking_at", "promise_at", "carrier_id",
    "warehouse_id", "destination_region", "service_level", "parcel_type",
    "weight_kg", "distance_km", "warehouse_backlog",
]
OUTCOME_COLUMNS = [
    "delivery_at", "delivered_on_time", "final_status", "exception_code",
    "customer_rating", "actual_cost_inr", "revised_promise_at",
]
CATEGORIES = ["carrier_id", "warehouse_id", "destination_region", "service_level", "parcel_type"]
NUMERIC = ["weight_kg", "distance_km", "warehouse_backlog", "promise_hours", "booking_age_hours", "warehouse_utilization"]
FEATURES = CATEGORIES + NUMERIC + ["dispatch_weekday", "is_weekend"]
REGIONS = {"NORTH", "SOUTH", "EAST", "WEST", "CENTRAL", "NORTHEAST"}
SERVICES = {"standard", "express"}
PARCEL_TYPES = {"standard", "fragile", "oversized"}


class InputValidationError(ValueError):
    """Actionable input contract violation."""


def schema_fingerprint() -> str:
    return hashlib.sha256(json.dumps(REQUIRED_INPUT, sort_keys=True).encode()).hexdigest()[:16]


def parse_aware(series: pd.Series, column: str) -> pd.Series:
    """Parse ISO timestamps; refuse naive values rather than guessing a timezone."""
    text = series.astype("string").str.strip()
    # Require an explicit Z or ±HH:MM offset. Do not let pandas silently infer one.
    has_zone = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    if (~has_zone & text.notna()).any():
        bad = text[~has_zone & text.notna()].iloc[0]
        raise InputValidationError(f"{column}: timestamp {bad!r} has no timezone offset")
    parsed = pd.to_datetime(text, format="mixed", utc=True, errors="coerce")
    if parsed.isna().any():
        raise InputValidationError(f"{column}: missing or invalid ISO-8601 timestamp")
    return parsed


def validate_input(df: pd.DataFrame, carriers: pd.DataFrame, warehouses: pd.DataFrame,
                   policy: dict[str, Any], *, strict_missing: bool = True) -> pd.DataFrame:
    """Validate the canonical inference contract and return a normalized copy.

    Training uses strict_missing=False: missing numeric/category values are retained,
    counted in the audit, and handled by a fitted preprocessing pipeline. Inference
    fails closed on missing values instead of silently imputing a live parcel.
    """
    if df.empty:
        raise InputValidationError("Input contains no parcels")
    if len(df) > 10_000:
        raise InputValidationError("Input exceeds 10,000 parcels; split the batch")
    missing = sorted(set(REQUIRED_INPUT) - set(df.columns))
    if missing:
        raise InputValidationError(f"Missing required columns: {missing}")
    out = df.copy()
    out["shipment_id"] = out["shipment_id"].astype("string").str.strip()
    if out["shipment_id"].isna().any() or (out["shipment_id"] == "").any():
        raise InputValidationError("shipment_id cannot be empty")
    if out["shipment_id"].duplicated().any():
        dup = out.loc[out["shipment_id"].duplicated(), "shipment_id"].iloc[0]
        raise InputValidationError(f"Duplicate shipment_id: {dup}")
    for col in ["snapshot_at", "booking_at", "promise_at"]:
        out[col] = parse_aware(out[col], col)
    local = out["snapshot_at"].dt.tz_convert(policy["timezone"])
    if (local.dt.hour != policy["prediction_local_hour"]).any() or (local.dt.minute != 0).any():
        raise InputValidationError("snapshot_at must be 06:00 Asia/Kolkata (00:30Z)")
    if (out["booking_at"] >= out["snapshot_at"]).any():
        raise InputValidationError("booking_at must be before snapshot_at")
    hours = (out["promise_at"] - out["snapshot_at"]).dt.total_seconds() / 3600
    if (hours <= 0).any():
        raise InputValidationError("promise_at must be after snapshot_at")
    if (hours > 24 * policy["max_promised_days"]).any():
        raise InputValidationError(f"promise_at exceeds {policy['max_promised_days']} days")
    for col in ["weight_kg", "distance_km", "warehouse_backlog"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if strict_missing and out[col].isna().any():
            raise InputValidationError(f"{col}: missing or non-numeric value")
        if (out[col] == np.inf).any() or (out[col] == -np.inf).any():
            raise InputValidationError(f"{col}: infinite value")
    limits = {"weight_kg": (0.01, 1000), "distance_km": (1, 10000), "warehouse_backlog": (0, 1_000_000)}
    for col, (lo, hi) in limits.items():
        bad = out[col].notna() & ~out[col].between(lo, hi)
        if bad.any():
            raise InputValidationError(f"{col}: values must be in [{lo}, {hi}]")
    if strict_missing and (out["warehouse_backlog"] % 1 != 0).any():
        raise InputValidationError("warehouse_backlog must be a whole number")
    for col in CATEGORIES:
        out[col] = out[col].astype("string").str.strip()
        if strict_missing and (out[col].isna().any() or (out[col] == "").any()):
            raise InputValidationError(f"{col}: missing category")
    out["carrier_id"] = out["carrier_id"].str.upper()
    out["warehouse_id"] = out["warehouse_id"].str.upper()
    out["destination_region"] = out["destination_region"].str.upper()
    out["service_level"] = out["service_level"].str.lower()
    out["parcel_type"] = out["parcel_type"].str.lower()
    for col, allowed in [("destination_region", REGIONS), ("service_level", SERVICES), ("parcel_type", PARCEL_TYPES)]:
        bad = out[col].notna() & ~out[col].isin(allowed)
        if bad.any():
            raise InputValidationError(f"{col}: unsupported value {out.loc[bad, col].iloc[0]!r}")
    carrier_ids = set(carriers["carrier_id"].str.upper())
    warehouse_ids = set(warehouses["warehouse_id"].str.upper())
    unknown_carrier = ~out["carrier_id"].isin(carrier_ids)
    unknown_warehouse = ~out["warehouse_id"].isin(warehouse_ids)
    if unknown_carrier.any():
        raise InputValidationError(f"Unknown carrier_id: {out.loc[unknown_carrier, 'carrier_id'].iloc[0]}")
    if unknown_warehouse.any():
        raise InputValidationError(f"Unknown warehouse_id: {out.loc[unknown_warehouse, 'warehouse_id'].iloc[0]}")
    # Effective-dated carrier contract must exist at the decision snapshot.
    ref = carriers.copy()
    ref["valid_from"] = pd.to_datetime(ref["valid_from"], utc=True)
    for row in out.itertuples():
        valid = ref[(ref["carrier_id"].str.upper() == row.carrier_id) & (ref["valid_from"] <= row.snapshot_at)]
        if valid.empty:
            raise InputValidationError(f"Carrier {row.carrier_id} has no contract effective at {row.snapshot_at}")
    return out


def engineer_features(df: pd.DataFrame, warehouses: pd.DataFrame) -> pd.DataFrame:
    """Only snapshot-known fields. Never reads delivery or other outcome columns."""
    out = df.copy()
    out["promise_hours"] = (out["promise_at"] - out["snapshot_at"]).dt.total_seconds() / 3600
    out["booking_age_hours"] = (out["snapshot_at"] - out["booking_at"]).dt.total_seconds() / 3600
    local = out["snapshot_at"].dt.tz_convert("Asia/Kolkata")
    out["dispatch_weekday"] = local.dt.weekday.astype(str)
    out["is_weekend"] = (local.dt.weekday >= 5).astype(int)
    cap = warehouses.set_index("warehouse_id")["daily_capacity"]
    capacity = out["warehouse_id"].map(cap).astype(float)
    out["warehouse_utilization"] = out["warehouse_backlog"] / capacity.replace(0, np.nan)
    return out[FEATURES]
