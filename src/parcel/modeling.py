from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import attach_contract
from .schema import CATEGORIES, FEATURES, NUMERIC, engineer_features

MODEL_VERSION = "parcel-0.1.0"


@dataclass
class Split:
    train: pd.DataFrame
    calibration: pd.DataFrame
    selection: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict


def temporal_split(df: pd.DataFrame) -> Split:
    """Chronological, disjoint shipment snapshots with an untouched forward update.

    Day 1 = 150 days; Day 2 update = next 30 days. Fit on days 0-119,
    calibrate on 120-134, select on 135-149, test only on 150-179.
    Shipments are disjoint; long promises can still overlap in calendar time,
    so this is not a perfect substitute for a production point-in-time backtest.
    """
    dates = np.array(sorted(df["snapshot_at"].dt.normalize().unique()))
    if len(dates) < 60:
        raise ValueError("Need at least 60 distinct snapshot dates for the configured temporal split")
    n = len(dates)
    if n >= 180:
        a, b, c = 120, 135, 150
    else:
        a, b, c = int(n * 0.67), int(n * 0.77), int(n * 0.84)
    day = df["snapshot_at"].dt.normalize()
    masks = [day < dates[a], (day >= dates[a]) & (day < dates[b]),
             (day >= dates[b]) & (day < dates[c]), day >= dates[c]]
    parts = [df[m].copy().reset_index(drop=True) for m in masks]
    # Label-maturity purge: a row may inform a decision only once its outcome is
    # known. Max synthetic promise is 5.5 days + 2 days late; an 8-day boundary
    # gap is enforced by dropping rows whose delivery crosses the next boundary.
    purged = {}
    for name, part, boundary in zip(["train", "calibration", "selection"], parts[:3], [dates[a], dates[b], dates[c]], strict=False):
        before = len(part)
        if "delivery_at" in part:
            part = part[part["delivery_at"].isna() | (part["delivery_at"] < boundary)].copy()
        parts[{"train": 0, "calibration": 1, "selection": 2}[name]] = part.reset_index(drop=True)
        purged[name] = before - len(part)
    if any(p.empty for p in parts):
        raise ValueError("Temporal split produced an empty partition")
    boundaries = {
        "train": [str(parts[0]["snapshot_at"].min()), str(parts[0]["snapshot_at"].max())],
        "calibration": [str(parts[1]["snapshot_at"].min()), str(parts[1]["snapshot_at"].max())],
        "selection": [str(parts[2]["snapshot_at"].min()), str(parts[2]["snapshot_at"].max())],
        "test": [str(parts[3]["snapshot_at"].min()), str(parts[3]["snapshot_at"].max())],
        "counts": {k: len(p) for k, p in zip(["train", "calibration", "selection", "test"], parts, strict=False)},
        "label_maturity_purged": purged,
        "purge_rule": "Drop fit/calibration/selection rows whose delivery_at is at or after the next partition start.",
    }
    return Split(*parts, boundaries)


def make_pipeline(kind: str) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", numeric, NUMERIC + ["is_weekend"]),
                             ("cat", categorical, CATEGORIES + ["dispatch_weekday"])],
                            remainder="drop")
    if kind == "logistic":
        estimator = LogisticRegression(C=0.5, max_iter=1500, random_state=42)
    elif kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=15, learning_rate=0.06,
            l2_regularization=1.0, min_samples_leaf=35, early_stopping=False, random_state=42)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("model", estimator)])


def baseline_probabilities(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Smoothed carrier x service historical late rate, fitted on training only."""
    labeled = train.dropna(subset=["late"])
    global_rate = float(labeled["late"].mean())
    grouped = labeled.groupby(["carrier_id", "service_level"])["late"].agg(["sum", "count"])
    rates = (grouped["sum"] + 20 * global_rate) / (grouped["count"] + 20)
    keys = pd.MultiIndex.from_frame(target[["carrier_id", "service_level"]])
    return np.asarray(rates.reindex(keys).fillna(global_rate), dtype=float)


def expected_benefit(prob: np.ndarray, cost: np.ndarray, policy: dict) -> np.ndarray:
    return prob * policy["intervention_success_probability"] * policy["missed_late_cost"] - cost


def allocate(frame: pd.DataFrame, policy: dict, warehouses: pd.DataFrame) -> pd.DataFrame:
    """Transparent greedy allocation by expected net benefit, with hard capacities.

    Greedy is not a proof of global optimality under overlapping carrier/warehouse
    constraints. It is deterministic, auditable, and never exceeds a limit.
    """
    out = frame.copy().reset_index(drop=True)
    out["flagged"] = False
    out["allocation_rank"] = pd.NA
    out["allocation_reason"] = "below economic threshold"
    wh_caps = warehouses.set_index("warehouse_id")["intervention_capacity"].to_dict()
    for _, idx in out.groupby(out["snapshot_at"].dt.normalize(), sort=True).groups.items():
        day = out.loc[idx]
        order = day.sort_values(["expected_net_benefit", "probability_late", "shipment_id"],
                                ascending=[False, False, True]).index
        total = 0
        carrier_counts: dict[str, int] = {}
        warehouse_counts: dict[str, int] = {}
        for i in order:
            row = out.loc[i]
            if row["expected_net_benefit"] <= 0:
                continue
            c, w = row["carrier_id"], row["warehouse_id"]
            if total >= policy["daily_capacity"]:
                out.at[i, "allocation_reason"] = "daily capacity reached"
                continue
            if carrier_counts.get(c, 0) >= policy["per_carrier_capacity"]:
                out.at[i, "allocation_reason"] = "carrier capacity reached"
                continue
            if warehouse_counts.get(w, 0) >= wh_caps.get(w, 0):
                out.at[i, "allocation_reason"] = "warehouse capacity reached"
                continue
            out.at[i, "flagged"] = True
            out.at[i, "allocation_rank"] = total + 1
            out.at[i, "allocation_reason"] = "positive expected benefit; capacity available"
            total += 1
            carrier_counts[c] = carrier_counts.get(c, 0) + 1
            warehouse_counts[w] = warehouse_counts.get(w, 0) + 1
    return out


def score_predictions(df: pd.DataFrame, prob: np.ndarray, policy: dict,
                      warehouses: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored = df.copy()
    scored["probability_late"] = np.clip(prob, 0, 1)
    scored["expected_net_benefit"] = expected_benefit(
        scored["probability_late"].to_numpy(), scored["intervention_cost_inr"].to_numpy(), policy)
    scored = allocate(scored, policy, warehouses)
    y = scored["late"].to_numpy(dtype=float)
    known = np.isfinite(y)
    flag = scored["flagged"].to_numpy()
    late = y == 1
    metrics: dict[str, Any] = {
        "n": int(known.sum()), "late_rate": float(y[known].mean()) if known.any() else None,
        "flagged": int(flag.sum()), "flag_rate": float(flag.mean()),
        "precision_at_capacity": float(y[flag].mean()) if flag.any() else 0.0,
        "recall_at_capacity": float((flag & late).sum() / late.sum()) if late.any() else None,
        "true_positive_flags": int((flag & late).sum()),
        "expected_net_benefit_inr": float(scored.loc[flag, "expected_net_benefit"].sum()),
        "intervention_cost_inr": float(scored.loc[flag, "intervention_cost_inr"].sum()),
        "expected_benefit_per_day_inr": float(scored.loc[flag, "expected_net_benefit"].sum() / max(1, scored["snapshot_at"].dt.normalize().nunique())),
    }
    if known.any() and len(np.unique(y[known])) == 2:
        metrics.update({
            "roc_auc": float(roc_auc_score(y[known], scored.loc[known, "probability_late"])),
            "pr_auc": float(average_precision_score(y[known], scored.loc[known, "probability_late"])),
            "brier": float(brier_score_loss(y[known], scored.loc[known, "probability_late"])),
        })
    return scored, metrics


def segment_report(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for col in ["carrier_id", "destination_region", "warehouse_id", "service_level", "parcel_type"]:
        for value, g in scored.groupby(col, observed=True):
            if len(g) < 30:
                continue
            late = g["late"] == 1
            flag = g["flagged"]
            rows.append({
                "segment": col, "value": str(value), "n": len(g),
                "late_rate": float(late.mean()),
                "recall_at_capacity": float((flag & late).sum() / late.sum()) if late.any() else None,
                "precision_at_capacity": float(g.loc[flag, "late"].mean()) if flag.any() else 0.0,
                "flagged": int(flag.sum()),
                "brier": float(brier_score_loss(g["late"], g["probability_late"])) if g["late"].nunique() == 2 else None,
            })
    return sorted(rows, key=lambda r: (r["recall_at_capacity"] is None, r["recall_at_capacity"] or 0))


def day_bootstrap(scored: pd.DataFrame, repetitions: int = 400, seed: int = 42) -> dict:
    """Resample whole decision days, not rows, for a rough stability interval."""
    days = []
    for _, g in scored.groupby(scored["snapshot_at"].dt.normalize()):
        days.append((int((g["flagged"] & g["late"].eq(1)).sum()), int(g["late"].eq(1).sum()),
                     float(g.loc[g["flagged"], "expected_net_benefit"].sum())))
    arr = np.array(days)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(arr), size=(repetitions, len(arr)))
    sums = arr[picks].sum(axis=1)
    recall = np.divide(sums[:, 0], sums[:, 1], out=np.full(repetitions, np.nan), where=sums[:, 1] > 0)
    benefit = sums[:, 2] / len(arr)
    return {
        "method": f"{repetitions} whole-day bootstrap resamples; synthetic data only",
        "recall_at_capacity_ci95": [float(np.nanpercentile(recall, 2.5)), float(np.nanpercentile(recall, 97.5))],
        "expected_benefit_per_day_ci95": [float(np.percentile(benefit, 2.5)), float(np.percentile(benefit, 97.5))],
    }


def reference_profile(features: pd.DataFrame) -> dict:
    profile: dict[str, Any] = {"n": len(features), "numeric": {}, "categorical": {}}
    for col in NUMERIC + ["dispatch_weekday", "is_weekend"]:
        s = pd.to_numeric(features[col], errors="coerce").dropna()
        edges = np.quantile(s, np.linspace(0, 1, 11)).tolist() if len(s) else []
        bin_counts = []
        if edges:
            bins = np.unique(np.array(edges, dtype=float))
            if len(bins) >= 2:
                bins[0], bins[-1] = -np.inf, np.inf
                bin_counts = np.histogram(s, bins=bins)[0].tolist()
        profile["numeric"][col] = {
            "quantiles": edges, "bin_counts": bin_counts,
            "missing_rate": float(features[col].isna().mean()),
            "median": float(s.median()) if len(s) else 0.0,
        }
    for col in CATEGORIES:
        profile["categorical"][col] = features[col].fillna("__MISSING__").value_counts(normalize=True).to_dict()
    return profile


def fit_and_evaluate(clean: pd.DataFrame, carriers: pd.DataFrame, warehouses: pd.DataFrame,
                     policy: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = temporal_split(clean)
    train = split.train.dropna(subset=["late"])
    X_train = engineer_features(train, warehouses)
    y_train = train["late"].astype(int)
    X_cal = engineer_features(split.calibration, warehouses)
    X_sel = engineer_features(split.selection, warehouses)
    X_test = engineer_features(split.test, warehouses)
    results: dict[str, Any] = {"model_version": MODEL_VERSION, "synthetic": True,
                               "split": split.boundaries, "policy": policy, "models": {}}
    candidates = {}
    # Day-1 anchor: fit on the same pre-update training window, evaluate on the
    # last 30 pre-update days. Retrospective anchor; not compared directly to the
    # shifted Day-2 test. The same-test baseline below is the fair comparison.
    test_start = split.test["snapshot_at"].min()
    pre = clean[clean["snapshot_at"] < test_start]
    anchor_start = test_start - pd.Timedelta(days=30)
    anchor_target = pre[pre["snapshot_at"] >= anchor_start].copy()
    anchor_prob = baseline_probabilities(train, anchor_target)
    _, anchor_metrics = score_predictions(anchor_target, anchor_prob, policy, warehouses)
    results["day1_baseline_pre_update"] = anchor_metrics
    for kind in ["logistic", "hist_gradient_boosting"]:
        base = make_pipeline(kind)
        base.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        calibrated.fit(X_cal, split.calibration["late"].astype(int))
        sel_prob = calibrated.predict_proba(X_sel)[:, 1]
        sel_scored, sel_metrics = score_predictions(split.selection, sel_prob, policy, warehouses)
        candidates[kind] = (calibrated, sel_metrics)
        results["models"][kind] = {"selection": sel_metrics}
    # Select by selection-window expected net benefit; tie-break by Brier.
    winner = max(candidates, key=lambda k: (candidates[k][1]["expected_net_benefit_inr"],
                                           -candidates[k][1].get("brier", 1)))
    model = candidates[winner][0]
    test_prob = model.predict_proba(X_test)[:, 1]
    test_scored, test_metrics = score_predictions(split.test, test_prob, policy, warehouses)
    baseline_test_prob = baseline_probabilities(train, split.test)
    _, baseline_test_metrics = score_predictions(split.test, baseline_test_prob, policy, warehouses)
    results["selected_model"] = winner
    results["selection_rule"] = "Highest selection-window expected net benefit under hard capacities; Brier tie-break. Test not used to choose."
    results["models"][winner]["test"] = test_metrics
    results["baseline_same_test"] = baseline_test_metrics
    results["test_uncertainty"] = day_bootstrap(test_scored)
    results["segments"] = segment_report(test_scored)
    # Global importance is measured on selection data, never on the test labels.
    sample = split.selection.sample(min(900, len(split.selection)), random_state=42)
    X_imp = engineer_features(sample, warehouses)
    imp = permutation_importance(model, X_imp, sample["late"].astype(int), scoring="roc_auc",
                                 n_repeats=3, random_state=42, n_jobs=1)
    importance = sorted([{"feature": f, "mean_auc_drop": float(m), "std": float(s)}
                         for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std, strict=False)],
                        key=lambda r: r["mean_auc_drop"], reverse=True)
    results["global_importance"] = importance
    profile = reference_profile(X_train)
    bundle = {
        "model": model, "carriers": carriers, "warehouses": warehouses, "policy": policy,
        "reference": profile, "model_version": MODEL_VERSION, "selected_model": winner,
        "feature_columns": FEATURES, "training_max_snapshot": str(train["snapshot_at"].max()),
        "global_importance": importance,
        "medians": {c: profile["numeric"][c]["median"] for c in NUMERIC if c in profile["numeric"]},
        "modes": {c: max(profile["categorical"][c], key=profile["categorical"][c].get) for c in CATEGORIES},
        "weekday_mode": str(X_train["dispatch_weekday"].mode().iloc[0]),
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    test_scored.to_csv(out_dir / "test_predictions.csv", index=False)
    # A canonical, outcome-free inference sample from the final update day.
    demo = split.test[split.test["snapshot_at"].dt.normalize() == split.test["snapshot_at"].dt.normalize().max()].copy()
    from .schema import REQUIRED_INPUT
    # The live inference contract rejects missing inputs. Export a complete-case
    # demo batch; do not silently relax validation just to make the demo pass.
    demo = demo.dropna(subset=REQUIRED_INPUT)
    demo[REQUIRED_INPUT].to_csv(out_dir / "demo_input.csv", index=False)
    demo_scored = test_scored[test_scored["shipment_id"].isin(demo["shipment_id"])].copy()
    demo_scored.to_csv(out_dir / "demo_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: parcel train --data-dir data/synthetic --out artifacts")
    bundle = joblib.load(path)
    if bundle.get("model_version") != MODEL_VERSION:
        raise ValueError("Incompatible model artifact version; retrain with this checkout")
    return bundle


def predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    from .schema import validate_input
    policy = bundle["policy"]
    validated = validate_input(df, bundle["carriers"], bundle["warehouses"], policy, strict_missing=True)
    validated = attach_contract(validated, bundle["carriers"])
    X = engineer_features(validated, bundle["warehouses"])
    prob = bundle["model"].predict_proba(X)[:, 1]
    validated["probability_late"] = prob
    validated["expected_net_benefit"] = expected_benefit(
        prob, validated["intervention_cost_inr"].to_numpy(), policy)
    validated = allocate(validated, policy, bundle["warehouses"])
    return validated
