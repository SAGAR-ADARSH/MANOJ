import pandas as pd

from parcel.modeling import load_bundle, predict
from parcel.monitoring import drift_check, outcome_check
from parcel.schema import validate_input


def test_drift_on_validated_demo_batch(trained, data_dir, policy):
    out, _ = trained
    bundle = load_bundle(out / "model.joblib")
    df = pd.read_csv(out / "demo_input.csv")
    validated = validate_input(df, bundle["carriers"], bundle["warehouses"], policy)
    report = drift_check(bundle, validated)
    assert report["status"] in {"ok", "warn"}
    assert report["n"] == len(df)
    assert any(c["check"] == "category:carrier_id" for c in report["checks"])


def test_unseen_carrier_triggers_warning(trained, valid_row):
    out, _ = trained
    bundle = load_bundle(out / "model.joblib")
    # CAR_D exists in the reference but not in the pre-update training window.
    valid_row["carrier_id"] = "CAR_D"
    valid_row["snapshot_at"] = "2025-06-10T00:30:00Z"
    valid_row["booking_at"] = "2025-06-09T00:30:00Z"
    valid_row["promise_at"] = "2025-06-13T00:30:00Z"
    df = validate_input(pd.DataFrame([valid_row]), bundle["carriers"], bundle["warehouses"], bundle["policy"])
    report = drift_check(bundle, df)
    carrier_check = next(c for c in report["checks"] if c["check"] == "category:carrier_id")
    assert carrier_check["unseen_rate"] == 1.0
    assert report["status"] == "warn"


def test_outcome_check_insufficient_data():
    report = outcome_check({"brier": 0.2, "recall_at_capacity": 0.5}, pd.DataFrame(columns=["late", "probability_late", "flagged"]))
    assert report["status"] == "insufficient_data"


def test_predict_public_path(trained, valid_row):
    out, _ = trained
    bundle = load_bundle(out / "model.joblib")
    scored = predict(bundle, pd.DataFrame([valid_row]))
    assert 0 <= scored.loc[0, "probability_late"] <= 1
    assert "flagged" in scored
    assert "expected_net_benefit" in scored
