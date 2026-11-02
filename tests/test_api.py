from fastapi.testclient import TestClient

from parcel.api import create_app


def test_health_and_summary(trained):
    out, _ = trained
    client = TestClient(create_app(out))
    assert client.get("/api/health").json()["synthetic"] is True
    summary = client.get("/api/summary").json()
    assert summary["selected_model"] in {"logistic", "hist_gradient_boosting"}
    assert summary["dates"]


def test_parcels_and_detail(trained):
    out, _ = trained
    client = TestClient(create_app(out))
    parcels = client.get("/api/parcels", params={"limit": 5}).json()["parcels"]
    assert parcels
    detail = client.get(f"/api/parcels/{parcels[0]['shipment_id']}").json()
    assert "explanation" in detail
    assert "caveat" in detail["explanation"]


def test_predict_rejects_bad_csv(trained):
    out, _ = trained
    client = TestClient(create_app(out))
    r = client.post("/api/predict", json={"csv": "shipment_id\nX1\n"})
    assert r.status_code == 422
    assert "Missing required columns" in r.json()["detail"]


def test_predict_valid_csv(trained):
    out, _ = trained
    client = TestClient(create_app(out))
    csv = (out / "demo_input.csv").read_text()
    r = client.post("/api/predict", json={"csv": csv})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    assert "drift" in body


def test_template_download(trained):
    out, _ = trained
    client = TestClient(create_app(out))
    r = client.get("/api/template")
    assert r.status_code == 200
    assert "shipment_id" in r.text
