from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CATEGORIES, NUMERIC, engineer_features


def drift_check(bundle: dict, df: pd.DataFrame) -> dict:
    """Concrete pre-label drift check using stored training bin counts.

    WARN does not silently become healthy. Small batches are flagged separately
    because a handful of parcels cannot establish a distribution shift.
    """
    X = engineer_features(df, bundle["warehouses"])
    ref = bundle["reference"]
    checks = []
    worst = 0.0
    small = len(df) < 30
    for col in NUMERIC:
        current = pd.to_numeric(X[col], errors="coerce").dropna().to_numpy()
        edges = ref["numeric"][col]["quantiles"]
        ref_counts = np.array(ref["numeric"][col].get("bin_counts", []), dtype=float)
        if len(edges) >= 3 and len(current) and len(ref_counts) >= 2:
            bins = np.unique(np.array(edges, dtype=float))
            bins[0], bins[-1] = -np.inf, np.inf
            counts = np.histogram(current, bins=bins)[0] + 1e-6
            ref_counts = ref_counts + 1e-6
            cur_p = counts / counts.sum()
            ref_p = ref_counts / ref_counts.sum()
            psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
        else:
            psi = 0.0
        worst = max(worst, psi)
        missing = float(X[col].isna().mean())
        checks.append({"check": f"numeric:{col}", "psi": round(psi, 4),
                       "missing_rate": round(missing, 4),
                       "reference_missing_rate": round(ref["numeric"][col]["missing_rate"], 4),
                       "status": "warn" if psi > 0.2 or missing > ref["numeric"][col]["missing_rate"] + 0.05 else "ok"})
    for col in CATEGORIES:
        current = X[col].fillna("__MISSING__").astype(str)
        known = set(ref["categorical"][col])
        unseen = float((~current.isin(known)).mean())
        worst = max(worst, unseen)
        checks.append({"check": f"category:{col}", "unseen_rate": round(unseen, 4),
                       "status": "warn" if unseen > 0.02 else "ok"})
    status = "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {
        "status": status, "n": len(df), "small_batch": small, "max_signal": round(worst, 4),
        "thresholds": {"numeric_psi_warn": 0.2, "unseen_category_rate_warn": 0.02, "min_batch_for_stability": 30},
        "checks": checks,
        "action": "Review carrier/warehouse mix and recent labeled outcomes before trusting the queue." if status == "warn" else "No configured input-drift alarm. This is not a guarantee of accuracy.",
        "limitation": "Input drift cannot detect every concept shift. Label-based recall and calibration checks are still required after outcomes mature. Batches under 30 are not stable evidence.",
    }


def outcome_check(reference_metrics: dict, labeled: pd.DataFrame) -> dict:
    """Post-outcome check against the frozen synthetic test anchor."""
    from sklearn.metrics import brier_score_loss
    required = {"late", "probability_late", "flagged"}
    if not required.issubset(labeled.columns):
        raise ValueError(f"Outcome monitoring needs columns: {sorted(required)}")
    g = labeled.dropna(subset=["late"])
    if len(g) < 30:
        return {"status": "insufficient_data", "n": len(g), "minimum_n": 30}
    brier = float(brier_score_loss(g["late"], g["probability_late"]))
    late = g["late"].eq(1)
    recall = float((g["flagged"] & late).sum()) / late.sum() if late.any() else None
    anchor_brier = reference_metrics.get("brier") or 0.0
    anchor_recall = reference_metrics.get("recall_at_capacity") or 0.0
    alarms = []
    if brier > anchor_brier + 0.03:
        alarms.append("Brier score worsened by more than 0.03")
    if recall is not None and recall < anchor_recall - 0.10:
        alarms.append("Recall at capacity fell by more than 10 percentage points")
    return {"status": "warn" if alarms else "ok", "n": len(g), "brier": brier,
            "recall_at_capacity": recall, "alarms": alarms,
            "caveat": "Thresholds are prototype defaults, not statistically calibrated production SLOs."}
