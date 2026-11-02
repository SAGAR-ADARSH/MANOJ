import pandas as pd
import pytest

from parcel.schema import InputValidationError, engineer_features, validate_input


def test_valid_row_passes(valid_row, references, policy):
    carriers, warehouses = references
    out = validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)
    assert len(out) == 1
    assert str(out["snapshot_at"].dt.tz) == "UTC"


def test_naive_timestamp_rejected(valid_row, references, policy):
    carriers, warehouses = references
    valid_row["snapshot_at"] = "2025-01-06T06:00:00"
    with pytest.raises(InputValidationError, match="timezone"):
        validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)


def test_duplicate_shipment_rejected(valid_row, references, policy):
    carriers, warehouses = references
    with pytest.raises(InputValidationError, match="Duplicate"):
        validate_input(pd.DataFrame([valid_row, valid_row]), carriers, warehouses, policy)


def test_missing_numeric_rejected_at_serving(valid_row, references, policy):
    carriers, warehouses = references
    valid_row["weight_kg"] = None
    with pytest.raises(InputValidationError, match="weight_kg"):
        validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)


def test_unknown_carrier_rejected(valid_row, references, policy):
    carriers, warehouses = references
    valid_row["carrier_id"] = "CAR_UNKNOWN"
    with pytest.raises(InputValidationError, match="Unknown carrier"):
        validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)


def test_impossible_weight_rejected(valid_row, references, policy):
    carriers, warehouses = references
    valid_row["weight_kg"] = -2
    with pytest.raises(InputValidationError, match="weight_kg"):
        validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)


def test_outcome_columns_are_not_features(valid_row, references, policy):
    carriers, warehouses = references
    valid_row["delivery_at"] = "2025-01-20T00:00:00Z"
    valid_row["delivered_on_time"] = 1
    df = validate_input(pd.DataFrame([valid_row]), carriers, warehouses, policy)
    features = engineer_features(df, warehouses)
    assert "delivery_at" not in features
    assert "delivered_on_time" not in features
    assert "late" not in features
