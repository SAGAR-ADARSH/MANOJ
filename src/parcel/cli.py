from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_policy
from .data import clean_shipments, load_references, write_audit
from .modeling import fit_and_evaluate, load_bundle, predict
from .monitoring import drift_check
from .schema import InputValidationError
from .synthetic import generate


def _write_update_diff(day1_audit: dict, combined_audit: dict, out: Path) -> None:
    diff = {
        "simulated": True,
        "new_clean_rows": combined_audit["clean_rows"] - day1_audit["clean_rows"],
        "day1_late_rate": day1_audit["late_rate"],
        "combined_late_rate": combined_audit["late_rate"],
        "changes": [
            "30 forward days and higher daily volume",
            "new carrier CAR_D with no training history",
            "pound-unit weights and carrier aliases",
            "mixed explicit timezone offsets",
            "CAR_C service deterioration",
        ],
        "response": "Re-audit, normalize units/aliases, retrain from raw, keep a forward test window, and surface unseen-carrier drift.",
    }
    (out / "update_diff.json").write_text(json.dumps(diff, indent=2))


def cmd_generate(args) -> int:
    manifest = generate(args.data_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_audit(args) -> int:
    policy = load_policy(args.policy)
    carriers, warehouses = load_references(args.data_dir)
    raw = pd.read_csv(args.data_dir / "day1_shipments.csv")
    if args.include_update:
        raw = pd.concat([raw, pd.read_csv(args.data_dir / "day2_update.csv")], ignore_index=True)
    clean, quarantine, audit = clean_shipments(raw, carriers, warehouses, policy)
    args.out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out / "clean.csv", index=False)
    quarantine.to_csv(args.out / "quarantine.csv", index=False)
    write_audit(audit, args.out)
    print(json.dumps({k: audit[k] for k in ["input_rows", "clean_rows", "quarantined_rows", "late_rate"]}, indent=2))
    return 0


def cmd_train(args) -> int:
    policy = load_policy(args.policy)
    carriers, warehouses = load_references(args.data_dir)
    day1_raw = pd.read_csv(args.data_dir / "day1_shipments.csv")
    day1, _, day1_audit = clean_shipments(day1_raw, carriers, warehouses, policy)
    update_path = args.data_dir / "day2_update.csv"
    if update_path.exists():
        raw = pd.concat([day1_raw, pd.read_csv(update_path)], ignore_index=True)
    else:
        raw = day1_raw
    clean, quarantine, audit = clean_shipments(raw, carriers, warehouses, policy)
    args.out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out / "clean.csv", index=False)
    quarantine.to_csv(args.out / "quarantine.csv", index=False)
    write_audit(audit, args.out)
    write_audit(day1_audit, args.out / "day1")
    _write_update_diff(day1_audit, audit, args.out)
    metrics = fit_and_evaluate(clean, carriers, warehouses, policy, args.out)
    print(json.dumps({"selected_model": metrics["selected_model"],
                      "test": metrics["models"][metrics["selected_model"]]["test"],
                      "baseline_same_test": metrics["baseline_same_test"]}, indent=2))
    return 0


def cmd_predict(args) -> int:
    bundle = load_bundle(args.model)
    df = pd.read_csv(args.input)
    scored = predict(bundle, df)
    from .explain import explain_rows
    flagged = scored[scored["flagged"]]
    explanations = explain_rows(bundle, flagged, max_rows=len(flagged))
    exp_by_id = {e["shipment_id"]: e for e in explanations}
    scored["reason"] = scored["shipment_id"].map(lambda x: exp_by_id.get(x, {}).get("summary", ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["shipment_id", "snapshot_at", "carrier_id", "warehouse_id", "probability_late",
            "expected_net_benefit", "flagged", "allocation_rank", "allocation_reason", "reason"]
    scored[cols].to_csv(args.output, index=False)
    print(json.dumps({"input": len(df), "flagged": int(scored["flagged"].sum()),
                      "output": str(args.output)}, indent=2))
    return 0


def cmd_monitor(args) -> int:
    bundle = load_bundle(args.model)
    df = pd.read_csv(args.input)
    from .schema import validate_input
    validated = validate_input(df, bundle["carriers"], bundle["warehouses"], bundle["policy"])
    report = drift_check(bundle, validated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "warn" else 0


def cmd_demo(args) -> int:
    generate(args.data_dir, seed=args.seed)
    args.include_update = True
    cmd_audit(args)
    cmd_train(args)
    args.model = args.out / "model.joblib"
    args.input = args.out / "demo_input.csv"
    args.output = args.out / "predictions.csv"
    cmd_predict(args)
    bundle = load_bundle(args.model)
    from .schema import validate_input
    demo_df = validate_input(pd.read_csv(args.input), bundle["carriers"], bundle["warehouses"], bundle["policy"])
    report = drift_check(bundle, demo_df)
    (args.out / "drift.json").write_text(json.dumps(report, indent=2))
    print(f"\nDemo ready. Start the dashboard with:\n  parcel serve --artifacts {args.out} --port 8000")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app
    app = create_app(args.artifacts)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parcel", description="Synthetic parcel delay decision desk")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Generate deterministic synthetic data")
    p.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seed", type=int, default=20250106)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("audit", help="Audit and clean historical data")
    p.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--include-update", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("train", help="Train, calibrate, select, and evaluate")
    p.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--policy", type=Path, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Score a canonical input CSV")
    p.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("monitor", help="Run input-drift checks; exit 2 on warning")
    p.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/drift.json"))
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("demo", help="Generate data, audit, train, predict, and monitor")
    p.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20250106)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="Serve the operations dashboard and API")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (InputValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
