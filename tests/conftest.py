from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel.config import load_policy  # noqa: E402
from parcel.data import clean_shipments, load_references  # noqa: E402
from parcel.modeling import fit_and_evaluate  # noqa: E402
from parcel.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def policy():
    return load_policy()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    generate(path, seed=20250106)
    return path


@pytest.fixture(scope="session")
def references(data_dir):
    return load_references(data_dir)


@pytest.fixture(scope="session")
def raw(data_dir):
    return pd.concat([
        pd.read_csv(data_dir / "day1_shipments.csv"),
        pd.read_csv(data_dir / "day2_update.csv"),
    ], ignore_index=True)


@pytest.fixture(scope="session")
def cleaned(raw, references, policy):
    carriers, warehouses = references
    return clean_shipments(raw, carriers, warehouses, policy)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, cleaned, references, policy):
    out = tmp_path_factory.mktemp("artifacts")
    clean, _, _ = cleaned
    carriers, warehouses = references
    metrics = fit_and_evaluate(clean, carriers, warehouses, policy, out)
    return out, metrics


@pytest.fixture()
def valid_row(data_dir):
    df = pd.read_csv(data_dir / "day1_shipments.csv")
    row = df.dropna(subset=["weight_kg", "distance_km"]).iloc[0]
    return {c: row[c] for c in [
        "shipment_id", "snapshot_at", "booking_at", "promise_at", "carrier_id",
        "warehouse_id", "destination_region", "service_level", "parcel_type",
        "weight_kg", "distance_km", "warehouse_backlog",
    ]}
