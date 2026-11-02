from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .explain import explain_rows
from .modeling import load_bundle, predict
from .monitoring import drift_check
from .schema import REQUIRED_INPUT, InputValidationError

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_CSV_BYTES = 2_000_000


class PredictRequest(BaseModel):
    csv: str = Field(..., min_length=1, max_length=MAX_CSV_BYTES)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", double_precision=6))


def create_app(artifacts: Path | str = "artifacts") -> FastAPI:
    artifacts = Path(artifacts)
    app = FastAPI(title="Parcel Decision Desk", version="0.1.0",
                  description="Synthetic demonstration. Not a production dispatch system.")
    state: dict[str, Any] = {}

    def bundle():
        if "bundle" not in state:
            state["bundle"] = load_bundle(artifacts / "model.joblib")
        return state["bundle"]

    def test_predictions() -> pd.DataFrame:
        if "test" not in state:
            path = artifacts / "test_predictions.csv"
            if not path.exists():
                raise HTTPException(503, "No scored test data. Run: parcel demo")
            df = pd.read_csv(path)
            for col in ["snapshot_at", "booking_at", "promise_at", "delivery_at"]:
                if col in df:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            state["test"] = df
        return state["test"]

    @app.exception_handler(InputValidationError)
    async def validation_error(_, exc):
        return JSONResponse(status_code=422, content={"error": "input_validation_failed", "detail": str(exc)})

    @app.get("/api/health")
    def health():
        b = bundle()
        return {"status": "ok", "synthetic": True, "model_version": b["model_version"],
                "selected_model": b["selected_model"], "training_max_snapshot": b["training_max_snapshot"]}

    @app.get("/api/summary")
    def summary():
        metrics = _read_json(artifacts / "metrics.json", {})
        audit = _read_json(artifacts / "audit.json", {})
        drift = _read_json(artifacts / "drift.json", {"status": "not_run"})
        b = bundle()
        df = test_predictions()
        dates = sorted(df["snapshot_at"].dt.strftime("%Y-%m-%d").unique().tolist())
        selected = metrics.get("selected_model")
        return {
            "synthetic": True, "policy": b["policy"], "metrics": metrics,
            "selected_model": selected, "test_metrics": metrics.get("models", {}).get(selected, {}).get("test", {}),
            "audit": {k: audit.get(k) for k in ["input_rows", "clean_rows", "quarantined_rows", "late_rate", "issues"]},
            "drift": drift, "dates": dates, "default_date": dates[-1] if dates else None,
            "global_importance": metrics.get("global_importance", []),
            "segments": metrics.get("segments", []),
            "baseline_same_test": metrics.get("baseline_same_test", {}),
            "day1_baseline_pre_update": metrics.get("day1_baseline_pre_update", {}),
            "test_uncertainty": metrics.get("test_uncertainty", {}),
        }

    @app.get("/api/parcels")
    def parcels(date: str | None = None, carrier: str | None = None,
                flagged_only: bool = False, q: str | None = None,
                limit: int = Query(500, ge=1, le=2000)):
        df = test_predictions()
        if date:
            df = df[df["snapshot_at"].dt.strftime("%Y-%m-%d") == date]
        if carrier:
            df = df[df["carrier_id"] == carrier]
        if flagged_only:
            df = df[df["flagged"] == True]  # noqa: E712
        if q:
            df = df[df["shipment_id"].str.contains(q, case=False, na=False)]
        df = df.sort_values(["flagged", "probability_late"], ascending=[False, False]).head(limit)
        return {"count": len(df), "parcels": _records(df)}

    @app.get("/api/parcels/{shipment_id}")
    def parcel_detail(shipment_id: str):
        df = test_predictions()
        row = df[df["shipment_id"] == shipment_id]
        if row.empty:
            raise HTTPException(404, "Shipment not found in the scored demonstration set")
        b = bundle()
        explanation = explain_rows(b, row, max_rows=1)[0]
        return {"parcel": _records(row)[0], "explanation": explanation}

    @app.post("/api/predict")
    def predict_csv(req: PredictRequest):
        if len(req.csv.encode("utf-8")) > MAX_CSV_BYTES:
            raise HTTPException(413, "CSV exceeds 2 MB")
        try:
            df = pd.read_csv(io.StringIO(req.csv))
        except Exception as exc:
            raise HTTPException(422, f"Malformed CSV: {exc}") from exc
        b = bundle()
        scored = predict(b, df)
        drift = drift_check(b, scored)
        flagged = scored[scored["flagged"]]
        explanations = explain_rows(b, flagged, max_rows=len(flagged))
        return {"count": len(scored), "flagged": int(scored["flagged"].sum()),
                "parcels": _records(scored), "drift": drift, "explanations": explanations}

    @app.get("/api/template", response_class=PlainTextResponse)
    def template():
        path = artifacts / "demo_input.csv"
        if path.exists():
            df = pd.read_csv(path).head(5)
            return df.to_csv(index=False)
        return ",".join(REQUIRED_INPUT) + "\n"

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
