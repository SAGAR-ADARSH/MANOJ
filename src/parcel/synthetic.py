"""Deterministic synthetic data. NOT real shipments and NOT a performance claim.

The generator deliberately plants the flaws documented in docs/DATA_AUDIT.md.
Day 2 is a forward-in-time update: a new carrier, pound units, carrier aliases,
mixed timezone offsets, duplicates, and a service deterioration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-01-06T06:00:00+05:30")
DAY1_DAYS = 150
DAY2_DAYS = 30


def _make_block(rng: np.random.Generator, start_day: int, days: int, per_day: int,
                prefix: str, update: bool) -> pd.DataFrame:
    rows = []
    carriers = ["CAR_A", "CAR_B", "CAR_C"] + (["CAR_D"] if update else [])
    carrier_p = [0.4, 0.35, 0.25] if not update else [0.3, 0.25, 0.2, 0.25]
    warehouses = ["WH_NORTH", "WH_SOUTH", "WH_CENTRAL"]
    regions = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL", "NORTHEAST"]
    carrier_effect = {"CAR_A": -0.55, "CAR_B": 0.05, "CAR_C": 0.5, "CAR_D": 1.0}
    for d in range(start_day, start_day + days):
        snapshot = START + pd.Timedelta(days=d)
        n = per_day + int(rng.integers(-12, 13))
        for i in range(n):
            carrier = str(rng.choice(carriers, p=carrier_p))
            warehouse = str(rng.choice(warehouses, p=[0.4, 0.35, 0.25]))
            region = str(rng.choice(regions, p=[0.22, 0.2, 0.16, 0.2, 0.15, 0.07]))
            service = str(rng.choice(["standard", "express"], p=[0.78, 0.22]))
            parcel_type = str(rng.choice(["standard", "fragile", "oversized"], p=[0.8, 0.14, 0.06]))
            weight = float(np.clip(rng.lognormal(0.7, 0.65), 0.2, 40))
            distance = float(np.clip(rng.gamma(4.0, 220), 30, 3500))
            backlog = int(rng.poisson({"WH_NORTH": 310, "WH_SOUTH": 250, "WH_CENTRAL": 190}[warehouse]))
            capacity = {"WH_NORTH": 400, "WH_SOUTH": 300, "WH_CENTRAL": 240}[warehouse]
            promise_days = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.35, 0.25, 0.15]))
            if service == "express":
                promise_days = max(1, promise_days - 2)
            promise = snapshot + pd.Timedelta(days=promise_days, hours=12)
            booking = snapshot - pd.Timedelta(hours=float(rng.uniform(2, 96)))
            util = backlog / capacity
            seasonal = 0.35 if 90 <= d < 120 else 0.0
            logit = (-2.05 + carrier_effect[carrier] + 0.00065 * distance
                     + 0.035 * weight + 1.1 * (util - 0.8)
                     + (0.45 if region == "NORTHEAST" else 0.0)
                     + (0.3 if parcel_type == "oversized" else 0.0)
                     + (-0.55 if service == "express" else 0.15)
                     + (0.25 if snapshot.dayofweek >= 5 else 0.0)
                     + seasonal + (0.45 if update and carrier == "CAR_C" else 0.0))
            p = 1 / (1 + np.exp(-logit))
            late = bool(rng.random() < p)
            if late:
                delivery = promise + pd.Timedelta(hours=float(rng.uniform(1, 48)))
            else:
                delivery = promise - pd.Timedelta(hours=float(rng.uniform(0.5, 36)))
            snap_utc = snapshot.tz_convert("UTC")
            weight_value = weight
            weight_unit = "kg"
            if update and rng.random() < 0.28:
                weight_value = weight * 2.2046226218
                weight_unit = "lb"
            carrier_raw = carrier
            if update and rng.random() < 0.08:
                carrier_raw = {"CAR_A": "car-a", "CAR_B": "Carrier B", "CAR_C": "car_c", "CAR_D": "car-d"}[carrier]
            snap_str = snap_utc.isoformat().replace("+00:00", "Z")
            if update and rng.random() < 0.35:
                snap_str = snapshot.isoformat()
            rows.append({
                "shipment_id": f"{prefix}-{d:03d}-{i:03d}",
                "snapshot_at": snap_str,
                "booking_at": booking.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
                "promise_at": promise.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
                "carrier_id": carrier_raw,
                "warehouse_id": warehouse,
                "destination_region": region,
                "service_level": service,
                "parcel_type": parcel_type,
                "weight_kg": round(weight_value, 3),
                "weight_unit": weight_unit,
                "distance_km": round(distance, 1),
                "warehouse_backlog": backlog,
                "delivery_at": delivery.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
                "delivered_on_time": int(not late),
                "final_status": "delivered",
                "exception_code": "DELAY" if late else "",
                "customer_rating": int(rng.integers(1, 4) if late else rng.integers(3, 6)),
                "actual_cost_inr": round(40 + 0.04 * distance + 3 * weight, 2),
                "revised_promise_at": (promise + pd.Timedelta(days=1)).isoformat() if late and rng.random() < 0.2 else "",
            })
    df = pd.DataFrame(rows)
    # Planted, documented quality issues. Rates are approximate by construction.
    n = len(df)
    miss = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[miss, "distance_km"] = np.nan
    miss = rng.choice(n, size=max(1, n // 150), replace=False)
    df.loc[miss, "weight_kg"] = np.nan
    bad = rng.choice(n, size=max(1, n // 400), replace=False)
    df.loc[bad, "weight_kg"] = -1.0
    # Exact duplicate rows.
    df = pd.concat([df, df.iloc[: max(1, n // 200)]], ignore_index=True)
    # Conflicting duplicate identity: no event version, so quarantine both.
    conflict = df.iloc[[10]].copy()
    conflict["distance_km"] = conflict["distance_km"] + 500
    df = pd.concat([df, conflict], ignore_index=True)
    return df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def generate(out_dir: Path | str, seed: int = 20250106) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    day1 = _make_block(rng, 0, DAY1_DAYS, 110, "S", False)
    day2 = _make_block(rng, DAY1_DAYS, DAY2_DAYS, 140, "S", True)
    day1.to_csv(out / "day1_shipments.csv", index=False)
    day2.to_csv(out / "day2_update.csv", index=False)
    carriers = pd.DataFrame([
        {"carrier_id": "CAR_A", "carrier_name": "Aster Express", "aliases": "car-a|Aster", "nominal_transit_days": 2, "intervention_cost_inr": 60, "valid_from": "2025-01-01T00:00:00Z"},
        {"carrier_id": "CAR_B", "carrier_name": "BlueRoute", "aliases": "Carrier B|blueroute", "nominal_transit_days": 3, "intervention_cost_inr": 75, "valid_from": "2025-01-01T00:00:00Z"},
        {"carrier_id": "CAR_C", "carrier_name": "Cedar Cargo", "aliases": "car_c|Cedar", "nominal_transit_days": 3, "intervention_cost_inr": 90, "valid_from": "2025-01-01T00:00:00Z"},
        {"carrier_id": "CAR_D", "carrier_name": "DeltaPost", "aliases": "car-d|Delta", "nominal_transit_days": 4, "intervention_cost_inr": 110, "valid_from": (START + pd.Timedelta(days=DAY1_DAYS)).tz_convert("UTC").isoformat().replace("+00:00", "Z")},
    ])
    carriers.to_csv(out / "carriers.csv", index=False)
    warehouses = pd.DataFrame([
        {"warehouse_id": "WH_NORTH", "daily_capacity": 400, "intervention_capacity": 35, "region": "NORTH"},
        {"warehouse_id": "WH_SOUTH", "daily_capacity": 300, "intervention_capacity": 30, "region": "SOUTH"},
        {"warehouse_id": "WH_CENTRAL", "daily_capacity": 240, "intervention_capacity": 25, "region": "CENTRAL"},
    ])
    warehouses.to_csv(out / "warehouses.csv", index=False)
    manifest = {
        "synthetic": True, "seed": seed,
        "warning": "Generated demonstration data. No real-company performance claim.",
        "day1": {"file": "day1_shipments.csv", "days": DAY1_DAYS, "approx_rows_per_day": 110},
        "day2_update": {"file": "day2_update.csv", "days": DAY2_DAYS, "approx_rows_per_day": 140,
                        "changes": ["new carrier CAR_D", "28% pound-unit weights", "carrier aliases", "mixed timezone offsets", "CAR_C deterioration", "higher volume"]},
        "day3": {"source": "config/policy.json", "simulated": True, "daily_capacity": 80, "per_carrier_capacity": 25},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
