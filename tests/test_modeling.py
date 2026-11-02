import numpy as np
import pandas as pd

from parcel.modeling import allocate, expected_benefit, temporal_split


def test_temporal_split_is_ordered_and_purged(cleaned):
    clean, _, _ = cleaned
    split = temporal_split(clean)
    assert split.train["snapshot_at"].max() < split.calibration["snapshot_at"].min()
    assert split.calibration["snapshot_at"].max() < split.selection["snapshot_at"].min()
    assert split.selection["snapshot_at"].max() < split.test["snapshot_at"].min()
    assert set(split.train["shipment_id"]).isdisjoint(split.test["shipment_id"])
    assert split.boundaries["label_maturity_purged"]["train"] > 0


def test_expected_benefit_threshold(policy):
    prob = np.array([0.0, 0.5, 1.0])
    cost = np.array([90.0, 90.0, 90.0])
    benefit = expected_benefit(prob, cost, policy)
    assert benefit[0] < 0
    assert benefit[2] == policy["missed_late_cost"] * policy["intervention_success_probability"] - 90


def test_allocation_respects_all_caps(policy):
    warehouses = pd.DataFrame([{"warehouse_id": "W1", "intervention_capacity": 2},
                               {"warehouse_id": "W2", "intervention_capacity": 2}])
    rows = []
    for i in range(12):
        rows.append({"shipment_id": f"P{i}", "snapshot_at": pd.Timestamp("2025-01-06T00:30:00Z"),
                     "carrier_id": "C1" if i < 6 else "C2",
                     "warehouse_id": "W1" if i % 2 else "W2",
                     "probability_late": 0.9, "expected_net_benefit": 100 - i})
    out = allocate(pd.DataFrame(rows), policy, warehouses)
    assert out["flagged"].sum() <= policy["daily_capacity"]
    assert out[out["flagged"]].groupby("carrier_id").size().max() <= policy["per_carrier_capacity"]
    assert out[out["flagged"]].groupby("warehouse_id").size().max() <= 2


def test_negative_benefit_never_flagged(policy):
    warehouses = pd.DataFrame([{"warehouse_id": "W1", "intervention_capacity": 10}])
    df = pd.DataFrame([{"shipment_id": "P1", "snapshot_at": pd.Timestamp("2025-01-06T00:30:00Z"),
                        "carrier_id": "C1", "warehouse_id": "W1", "probability_late": 0.9,
                        "expected_net_benefit": -1}])
    assert not allocate(df, policy, warehouses)["flagged"].any()


def test_demo_input_is_complete_case(trained):
    out, _ = trained
    demo = pd.read_csv(out / "demo_input.csv")
    assert not demo.isna().any().any()
    assert len(demo) > 0
