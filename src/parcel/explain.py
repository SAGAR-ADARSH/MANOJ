from __future__ import annotations

import pandas as pd

from .schema import FEATURES, NUMERIC, engineer_features

LABELS = {
    "carrier_id": "Carrier", "warehouse_id": "Warehouse", "destination_region": "Destination region",
    "service_level": "Service level", "parcel_type": "Parcel type",
    "weight_kg": "Parcel weight (kg)", "distance_km": "Planned distance (km)",
    "warehouse_backlog": "Warehouse backlog at 06:00", "promise_hours": "Hours left to original promise",
    "booking_age_hours": "Hours since booking", "warehouse_utilization": "Backlog / daily capacity",
    "dispatch_weekday": "Dispatch weekday", "is_weekend": "Weekend dispatch",
}


def _value_text(feature: str, value) -> str:
    if feature in NUMERIC:
        if feature in {"promise_hours", "booking_age_hours"}:
            return f"{float(value):.0f} hours"
        if feature == "warehouse_utilization":
            return f"{float(value):.0%} of daily capacity"
        return f"{float(value):,.1f}"
    return str(value)


def explain_rows(bundle: dict, rows: pd.DataFrame, max_rows: int = 100) -> list[dict]:
    """Model-agnostic local sensitivity around a training reference.

    These are not causal effects, not SHAP values, and not additive contributions.
    Correlated features are intentionally left unchanged; the caveat is surfaced.
    """
    rows = rows.head(max_rows).copy()
    if rows.empty:
        return []
    X = engineer_features(rows, bundle["warehouses"])[FEATURES]
    base = bundle["model"].predict_proba(X)[:, 1]
    explanations = []
    for pos, (_, row) in enumerate(rows.iterrows()):
        drivers = []
        variants = []
        names = []
        for feature in FEATURES:
            if feature == "dispatch_weekday":
                ref = bundle.get("weekday_mode", "0")
            elif feature == "is_weekend":
                ref = 0
            else:
                ref = bundle["modes"].get(feature, bundle["medians"].get(feature))
            if pd.isna(X.iloc[pos][feature]) or X.iloc[pos][feature] == ref:
                continue
            variant = X.iloc[[pos]].copy()
            variant[feature] = ref
            variants.append(variant)
            names.append(feature)
        if variants:
            all_variants = pd.concat(variants, ignore_index=True)
            alt = bundle["model"].predict_proba(all_variants)[:, 1]
            for feature, _changed, p_alt in zip(names, variants, alt, strict=False):
                delta = float(base[pos] - p_alt)
                if delta > 0.001:
                    drivers.append({
                        "feature": feature,
                        "label": LABELS.get(feature, feature),
                        "value": _value_text(feature, X.iloc[pos][feature]),
                        "risk_increase": round(delta, 4),
                        "text": f"{LABELS.get(feature, feature)} is {_value_text(feature, X.iloc[pos][feature])}; replacing it with a typical training value lowers estimated late risk by {delta:.1%}.",
                    })
        drivers.sort(key=lambda d: d["risk_increase"], reverse=True)
        explanations.append({
            "shipment_id": str(row["shipment_id"]),
            "probability_late": round(float(base[pos]), 4),
            "drivers": drivers[:3],
            "summary": drivers[0]["text"] if drivers else "No single input stands out against the training reference; the combination of inputs drives the estimate.",
            "caveat": "Local sensitivity, not a causal or additive explanation. Correlated inputs are not adjusted together.",
        })
    return explanations
