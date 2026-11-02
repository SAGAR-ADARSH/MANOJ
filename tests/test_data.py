import pandas as pd

from parcel.data import attach_contract, clean_shipments


def test_audit_finds_planted_issues(cleaned):
    _, quarantine, audit = cleaned
    issues = {i["issue"] for i in audit["issues"]}
    assert "Exact duplicate rows" in issues
    assert "Pounds mixed with kilograms" in issues
    assert "Carrier aliases / case" in issues
    assert "Mixed timezone offsets" in issues
    assert len(quarantine) > 0
    assert audit["clean_rows"] + audit["quarantined_rows"] <= audit["input_rows"]


def test_pounds_converted(references, policy):
    carriers, warehouses = references
    base = pd.DataFrame([{
        "shipment_id": "X1", "snapshot_at": "2025-01-06T00:30:00Z",
        "booking_at": "2025-01-05T00:30:00Z", "promise_at": "2025-01-09T00:30:00Z",
        "carrier_id": "CAR_A", "warehouse_id": "WH_NORTH", "destination_region": "NORTH",
        "service_level": "standard", "parcel_type": "standard", "weight_kg": 10.0,
        "weight_unit": "lb", "distance_km": 100, "warehouse_backlog": 10,
        "delivery_at": "2025-01-10T00:30:00Z", "final_status": "delivered",
    }])
    clean, _, _ = clean_shipments(base, carriers, warehouses, policy)
    assert abs(clean.loc[0, "weight_kg"] - 4.5359237) < 1e-6


def test_conflicting_duplicate_quarantined(references, policy):
    carriers, warehouses = references
    row = {
        "shipment_id": "X1", "snapshot_at": "2025-01-06T00:30:00Z",
        "booking_at": "2025-01-05T00:30:00Z", "promise_at": "2025-01-09T00:30:00Z",
        "carrier_id": "CAR_A", "warehouse_id": "WH_NORTH", "destination_region": "NORTH",
        "service_level": "standard", "parcel_type": "standard", "weight_kg": 2,
        "distance_km": 100, "warehouse_backlog": 10,
        "delivery_at": "2025-01-10T00:30:00Z", "final_status": "delivered",
    }
    other = dict(row, distance_km=999)
    clean, quarantine, _ = clean_shipments(pd.DataFrame([row, other]), carriers, warehouses, policy)
    assert clean.empty
    assert len(quarantine) == 2


def test_label_uses_original_promise(references, policy):
    carriers, warehouses = references
    row = {
        "shipment_id": "X1", "snapshot_at": "2025-01-06T00:30:00Z",
        "booking_at": "2025-01-05T00:30:00Z", "promise_at": "2025-01-09T00:30:00Z",
        "carrier_id": "CAR_A", "warehouse_id": "WH_NORTH", "destination_region": "NORTH",
        "service_level": "standard", "parcel_type": "standard", "weight_kg": 2,
        "distance_km": 100, "warehouse_backlog": 10,
        "delivery_at": "2025-01-10T00:30:00Z", "final_status": "delivered",
        "delivered_on_time": 1, "revised_promise_at": "2025-01-11T00:30:00Z",
    }
    clean, _, audit = clean_shipments(pd.DataFrame([row]), carriers, warehouses, policy)
    assert clean.loc[0, "late"] == 1
    assert any("Supplied on-time flag" in i["issue"] for i in audit["issues"])


def test_contract_join_accepts_validated_string_dtype(valid_row, references, policy):
    from parcel.schema import validate_input
    carriers, _ = references
    df = validate_input(pd.DataFrame([valid_row]), carriers, references[1], policy)
    out = attach_contract(df, carriers)
    assert out["intervention_cost_inr"].notna().all()
