from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import CATEGORIES, OUTCOME_COLUMNS, REQUIRED_INPUT, parse_aware

COLUMN_AVAILABILITY = {
    "shipment_id": ("Booking", "Identifier only; never a model feature"),
    "snapshot_at": ("Decision time", "Defines the 06:00 IST prediction cutoff"),
    "booking_at": ("Booking", "Booking age is known at the cutoff"),
    "promise_at": ("Booking", "Original customer promise; define the label against this"),
    "carrier_id": ("Booking / contract", "Known before dispatch; normalize aliases"),
    "warehouse_id": ("Booking", "Known before dispatch"),
    "destination_region": ("Booking", "Known before dispatch"),
    "service_level": ("Booking", "Purchased service is known"),
    "parcel_type": ("Booking", "Known before dispatch"),
    "weight_kg": ("Booking / intake", "Must be measured before cutoff; normalize units"),
    "weight_unit": ("Intake", "Explicit unit; missing means kg only in this synthetic contract"),
    "distance_km": ("Booking / route planning", "Planned road distance, not post-delivery distance"),
    "warehouse_backlog": ("06:00 operational snapshot", "Must be an as-of count, not end-of-day backlog"),
    "delivery_at": ("After outcome", "EXCLUDE: directly determines the label"),
    "delivered_on_time": ("After outcome", "EXCLUDE: target-derived field; may use a moving promise"),
    "final_status": ("After outcome", "EXCLUDE: unavailable for open parcels"),
    "exception_code": ("During / after delivery", "EXCLUDE: not reliably known at decision time"),
    "customer_rating": ("After delivery", "EXCLUDE: post-outcome feedback"),
    "actual_cost_inr": ("After billing", "EXCLUDE: use contracted intervention cost instead"),
    "revised_promise_at": ("After booking, often after delay", "EXCLUDE: changing the promise can hide lateness"),
}


def load_references(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    carriers = pd.read_csv(data_dir / "carriers.csv", dtype={"carrier_id": str})
    warehouses = pd.read_csv(data_dir / "warehouses.csv", dtype={"warehouse_id": str})
    if carriers["carrier_id"].duplicated().any():
        raise ValueError("Carrier reference must have one row per carrier in this prototype")
    if warehouses["warehouse_id"].duplicated().any():
        raise ValueError("Duplicate warehouse_id in capacity reference")
    for col in ["nominal_transit_days", "intervention_cost_inr"]:
        carriers[col] = pd.to_numeric(carriers[col], errors="raise")
        if (carriers[col] <= 0).any():
            raise ValueError(f"Carrier reference: {col} must be positive")
    for col in ["daily_capacity", "intervention_capacity"]:
        warehouses[col] = pd.to_numeric(warehouses[col], errors="raise")
        if (warehouses[col] <= 0).any() or (warehouses[col] % 1 != 0).any():
            raise ValueError(f"Warehouse reference: {col} must be a positive integer")
    carriers["valid_from"] = parse_aware(carriers["valid_from"], "valid_from")
    return carriers, warehouses


def attach_contract(df: pd.DataFrame, carriers: pd.DataFrame) -> pd.DataFrame:
    """Use the latest carrier contract effective at each snapshot, never a future cost."""
    left = df[["shipment_id", "snapshot_at", "carrier_id"]].copy().reset_index(drop=True)
    right = carriers[["carrier_id", "valid_from", "intervention_cost_inr", "nominal_transit_days"]].copy()
    # pandas nullable string vs object keys are not merge-compatible in merge_asof.
    left["carrier_id"] = left["carrier_id"].astype(str)
    right["carrier_id"] = right["carrier_id"].astype(str)
    left = left.sort_values("snapshot_at")
    right = right.sort_values("valid_from")
    merged = pd.merge_asof(left, right, left_on="snapshot_at", right_on="valid_from",
                           by="carrier_id", direction="backward")
    if merged["intervention_cost_inr"].isna().any():
        raise ValueError("A shipment has no effective carrier contract")
    return df.merge(merged[["shipment_id", "intervention_cost_inr", "nominal_transit_days"]],
                    on="shipment_id", how="left", validate="one_to_one")


def clean_shipments(raw: pd.DataFrame, carriers: pd.DataFrame, warehouses: pd.DataFrame,
                    policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit and clean historical data. Quarantine is explicit, never silent row loss."""
    issues: list[dict] = []
    df = raw.copy()
    initial_rows = len(df)
    exact = int(df.duplicated().sum())
    if exact:
        issues.append({"issue": "Exact duplicate rows", "count": exact, "action": "Drop identical copies; keep one event"})
    df = df.drop_duplicates().reset_index(drop=True)
    # Normalize identities using the reference alias table.
    alias_map = {}
    for row in carriers.itertuples():
        alias_map[row.carrier_id.upper()] = row.carrier_id
        for alias in str(row.aliases).split("|"):
            alias_map[alias.strip().upper()] = row.carrier_id
    original_carrier = df["carrier_id"].astype("string").str.strip().str.upper()
    df["carrier_id"] = original_carrier.map(alias_map).fillna(original_carrier)
    aliases = int((original_carrier != df["carrier_id"]).sum())
    if aliases:
        issues.append({"issue": "Carrier aliases / case", "count": aliases, "action": "Map through carrier reference aliases"})
    for col in ["warehouse_id", "destination_region"]:
        df[col] = df[col].astype("string").str.strip().str.upper()
    for col in ["service_level", "parcel_type"]:
        df[col] = df[col].astype("string").str.strip().str.lower()
    # Explicit units. The synthetic data contract says blank unit = kg; real data must confirm this.
    if "weight_unit" not in df:
        df["weight_unit"] = "kg"
        issues.append({"issue": "Missing weight_unit column", "count": initial_rows, "action": "Synthetic contract: assume kg; confirm before real use"})
    unit = df["weight_unit"].fillna("kg").astype(str).str.strip().str.lower()
    pounds = unit.eq("lb") | unit.eq("lbs")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df.loc[pounds, "weight_kg"] = df.loc[pounds, "weight_kg"] * 0.45359237
    bad_unit = ~unit.isin(["kg", "lb", "lbs"])
    if bad_unit.any():
        issues.append({"issue": "Unknown weight unit", "count": int(bad_unit.sum()), "action": "Quarantine; never guess units"})
    df.loc[pounds, "weight_unit"] = "kg_converted_from_lb"
    df.loc[~pounds, "weight_unit"] = "kg"
    if pounds.any():
        issues.append({"issue": "Pounds mixed with kilograms", "count": int(pounds.sum()), "action": "Convert lb to kg using 0.45359237"})
    # Timestamp offsets are mandatory. Mixed offsets are normalized, not stripped.
    tz_counts = {}
    for col in ["snapshot_at", "booking_at", "promise_at", "delivery_at"]:
        if col not in df:
            continue
        text = df[col].astype("string").str.strip()
        tz_counts[col] = int(text.str.endswith("+05:30").sum())
        try:
            df[col] = parse_aware(text, col)
        except ValueError:
            # Row-level quarantine for historical data; inference uses the strict parser.
            has_zone = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
            df[col] = pd.to_datetime(text.where(has_zone), format="mixed", utc=True, errors="coerce")
            issues.append({"issue": f"Invalid/naive {col}", "count": int(df[col].isna().sum()), "action": "Quarantine rows; require explicit offsets"})
    mixed = sum(tz_counts.values())
    if mixed:
        issues.append({"issue": "Mixed timezone offsets", "count": mixed, "action": "Parse offsets and normalize to UTC; derive business day in IST"})
    # Conflicting identities have no trustworthy version ordering.
    canonical_cols = [c for c in REQUIRED_INPUT if c in df]
    conflict_ids = df.groupby("shipment_id")[canonical_cols].nunique(dropna=False).gt(1).any(axis=1)
    conflict_ids = set(conflict_ids[conflict_ids].index)
    reasons = pd.Series("", index=df.index, dtype="string")
    def mark(mask, reason):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + reason + "; ")
    mark(df["shipment_id"].isin(conflict_ids), "conflicting duplicate identity")
    mark(bad_unit, "unknown weight unit")
    mark(df["weight_kg"].notna() & ~df["weight_kg"].between(0.01, 1000), "impossible weight")
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    mark(df["distance_km"].notna() & ~df["distance_km"].between(1, 10000), "impossible distance")
    df["warehouse_backlog"] = pd.to_numeric(df["warehouse_backlog"], errors="coerce")
    mark(df["warehouse_backlog"].notna() & ~df["warehouse_backlog"].between(0, 1_000_000), "impossible backlog")
    mark(df["booking_at"].isna() | df["snapshot_at"].isna() | df["promise_at"].isna(), "invalid required timestamp")
    mark(df["booking_at"] >= df["snapshot_at"], "booking not before cutoff")
    mark(df["promise_at"] <= df["snapshot_at"], "promise already passed at cutoff")
    mark(~df["carrier_id"].isin(set(carriers["carrier_id"])), "unknown carrier")
    mark(~df["warehouse_id"].isin(set(warehouses["warehouse_id"])), "unknown warehouse")
    if "delivery_at" in df:
        mark(df["delivery_at"].notna() & (df["delivery_at"] <= df["snapshot_at"]), "delivery not after cutoff")
    quarantined = df[reasons != ""].copy()
    quarantined["quarantine_reason"] = reasons[reasons != ""]
    for reason, count in quarantined["quarantine_reason"].value_counts().items():
        issues.append({"issue": reason.strip("; "), "count": int(count), "action": "Quarantine; exclude from training/evaluation until corrected"})
    clean = df[reasons == ""].copy().reset_index(drop=True)
    # Missingness is retained for training; the fitted pipeline imputes it.
    for col in ["weight_kg", "distance_km", "warehouse_backlog"] + CATEGORIES:
        n = int(clean[col].isna().sum())
        if n:
            issues.append({"issue": f"Missing {col}", "count": n, "action": "Training: fitted imputation + indicator (numeric); inference: reject missing"})
    if "delivery_at" in clean:
        resolved = clean["delivery_at"].notna() & clean["final_status"].eq("delivered")
        clean["late"] = np.where(resolved, (clean["delivery_at"] > clean["promise_at"]).astype(int), np.nan)
        unresolved = int((~resolved).sum())
        if unresolved:
            issues.append({"issue": "Unresolved delivery outcome", "count": unresolved, "action": "Exclude from labeled evaluation; never call it on-time"})
        if "delivered_on_time" in clean:
            supplied = pd.to_numeric(clean["delivered_on_time"], errors="coerce")
            disagreements = int((resolved & supplied.notna() & ((1 - supplied) != clean["late"])).sum())
            if disagreements:
                issues.append({"issue": "Supplied on-time flag disagrees with original promise", "count": disagreements, "action": "Recompute label from delivery_at > original promise_at"})
    clean = attach_contract(clean, carriers)
    audit = {
        "synthetic": True, "input_rows": initial_rows, "clean_rows": len(clean),
        "quarantined_rows": len(quarantined), "issues": issues,
        "date_min": str(clean["snapshot_at"].min()), "date_max": str(clean["snapshot_at"].max()),
        "late_rate": float(clean["late"].mean()) if "late" in clean else None,
        "column_availability": {c: {"known_at": a, "decision": d} for c, (a, d) in COLUMN_AVAILABILITY.items() if c in raw},
        "outcome_columns_excluded": [c for c in OUTCOME_COLUMNS if c in raw],
    }
    return clean, quarantined, audit


def write_audit(audit: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    lines = ["# Data audit — synthetic demonstration", "",
             "> Generated data, not evidence about a real carrier or company.", "",
             f"- Input rows: **{audit['input_rows']:,}**",
             f"- Clean rows: **{audit['clean_rows']:,}**",
             f"- Quarantined rows: **{audit['quarantined_rows']:,}**",
             f"- Snapshot range: {audit['date_min']} to {audit['date_max']}",
             f"- Observed late rate: {audit['late_rate']:.2%}" if audit['late_rate'] is not None else "",
             "", "## Issues and decisions", "", "| Issue | Rows | Decision |", "|---|---:|---|"]
    lines += [f"| {i['issue']} | {i['count']} | {i['action']} |" for i in audit["issues"]]
    lines += ["", "## Column-by-column availability", "", "| Column | Known at | Decision |", "|---|---|---|"]
    lines += [f"| `{c}` | {v['known_at']} | {v['decision']} |" for c, v in audit["column_availability"].items()]
    lines += ["", "## Limits of this audit", "",
              "The generator plants known flaws so the cleaning path can be tested. A real extract may contain additional issues. ",
              "Carrier aliases are resolved only when listed in the reference table. Conflicting shipment identities are quarantined, not guessed. ",
              "Missing numeric values are imputed only for training; live inference fails closed. ",
              "The backlog field is assumed to be a true 06:00 snapshot; a real feed must prove its as-of semantics."]
    (out_dir / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
