# Parcel · Decision Desk

**Project selection:** I chose **Project Parcel** because it lets me show the part I care about most: turning messy operational data into a decision system whose evaluation, cost assumptions, and failure behavior are visible rather than hidden behind a score.

> **Synthetic prototype.** This repository contains generated shipments, fictional carriers, and explicitly simulated Day 2/3 updates. It is a complete working system for the assignment structure, **not** evidence of real-company performance and **not** a production dispatch system.

## What is included

- Deterministic synthetic shipment generator with planted data flaws.
- Historical audit, unit/alias/timezone normalization, explicit quarantine, and column-level leakage decisions.
- Late label: `delivery_at > original promise_at`, evaluated at a **06:00 Asia/Kolkata** decision snapshot.
- Two meaningfully different models: regularized logistic regression and histogram gradient boosting, both sigmoid-calibrated.
- Chronological fit/calibration/selection/test split with label-maturity purging; the Day 2 update is the forward test window.
- Cost-aware, capacity-constrained allocation: daily, per-carrier, and per-warehouse hard caps.
- CLI and FastAPI service with strict input validation.
- Operations dashboard: ranked queue, parcel explanations, model comparison, weak segments, drift monitor, audit, and CSV upload.
- Input-drift monitoring and a post-outcome monitoring function.
- Automated tests, failure log, draft check-ins, decision memo, walkthrough script, and a generated ≤4-page PDF report.

## Quick start (clean checkout)

Requires **Python 3.11+**. No external model API or network access is needed after dependency installation.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Generate data, audit, train, evaluate, score a demo batch, and run drift checks.
parcel demo

# Start the dashboard and API.
parcel serve --artifacts artifacts --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. In the Arena live preview, use the provided forwarded port instead.

The first `parcel demo` takes roughly 10–30 seconds on a laptop. It writes reproducible data to `data/synthetic/` and models/reports to `artifacts/`. Both directories are gitignored; regenerate them from a clean checkout.

## Common commands

```bash
# Regenerate only the synthetic assets.
parcel generate --data-dir data/synthetic --seed 20250106

# Audit Day 1 only, or Day 1 + simulated update.
parcel audit --data-dir data/synthetic --out artifacts/day1
parcel audit --data-dir data/synthetic --out artifacts --include-update

# Train and evaluate. Uses config/policy.json for costs and capacities.
parcel train --data-dir data/synthetic --out artifacts --policy config/policy.json

# Score a canonical CSV. Fails clearly on bad input.
parcel predict --model artifacts/model.joblib \
  --input artifacts/demo_input.csv --output artifacts/predictions.csv

# Input-drift check. Exit 0 = no configured alarm; exit 2 = warning.
parcel monitor --model artifacts/model.joblib --input artifacts/demo_input.csv

# Tests and lint.
pytest
ruff check src tests

# Rebuild the submission report and evaluation notes from current artifacts.
python scripts/build_report.py
```

## Input contract

`parcel predict` and `POST /api/predict` expect a canonical CSV with these columns:

| Column | Rule |
|---|---|
| `shipment_id` | Non-empty, unique within the batch |
| `snapshot_at` | ISO-8601 with explicit offset; must be 06:00 Asia/Kolkata |
| `booking_at` | ISO-8601 with offset; before snapshot |
| `promise_at` | Original promise; after snapshot, within 14 days |
| `carrier_id` | Known in `carriers.csv`, with a contract effective at snapshot |
| `warehouse_id` | Known in `warehouses.csv` |
| `destination_region` | NORTH, SOUTH, EAST, WEST, CENTRAL, NORTHEAST |
| `service_level` | standard or express |
| `parcel_type` | standard, fragile, oversized |
| `weight_kg` | 0.01–1000 kg; no missing values at serving time |
| `distance_km` | 1–10000 km; no missing values at serving time |
| `warehouse_backlog` | Whole number, 0–1,000,000; true 06:00 snapshot |

Historical training tolerates documented missingness and fits imputers. **Live inference rejects missing required values** rather than silently imputing a customer parcel. Raw pound units and carrier aliases are handled in the historical audit path; the canonical serving contract expects kilograms and canonical IDs.

Download a five-row template from `GET /api/template` or use `artifacts/demo_input.csv` after `parcel demo`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model version and synthetic flag |
| `GET /api/summary` | Metrics, policy, audit, drift, segments, importance |
| `GET /api/parcels?date=YYYY-MM-DD&carrier=CAR_A&flagged_only=true` | Ranked historical demo queue |
| `GET /api/parcels/{shipment_id}` | Parcel detail + local explanation |
| `POST /api/predict` with `{"csv": "..."}` | Validate, score, allocate, explain, drift-check a new batch (≤2 MB) |
| `GET /api/template` | Example canonical CSV |

Interactive API docs: `http://localhost:8000/docs`.

## Decision logic

For a parcel with calibrated late probability `p`, carrier intervention cost `c`, assumed intervention success `s = 0.40`, and assumed avoided late cost `L = ₹450`:

```text
expected_net_benefit = p × s × L − c
```

Only positive-benefit parcels are eligible. Eligible parcels are ranked by expected net benefit and allocated greedily subject to:

- **80 interventions/day** total (simulated Day 3 constraint),
- **25 interventions/day/carrier**,
- warehouse caps from `warehouses.csv` (35 / 30 / 25).

The economic threshold is `p > c / (s × L)`: about 0.33 for CAR_A, 0.42 for CAR_B, 0.50 for CAR_C, and 0.61 for CAR_D. In the current synthetic run, economics—not capacity—limits interventions. That is a finding, not a bug: the system refuses to spend money on parcels whose estimated benefit does not cover cost.

Greedy allocation is deterministic and auditable, but not proven globally optimal under overlapping carrier/warehouse constraints.

## Repository map

```text
config/policy.json          Simulated Day 3 cost/capacity policy
src/parcel/synthetic.py     Deterministic data generator
src/parcel/data.py          Audit, cleaning, quarantine, effective-dated contracts
src/parcel/schema.py        Serving contract and feature engineering
src/parcel/modeling.py      Split, models, calibration, allocation, evaluation
src/parcel/explain.py       Local sensitivity explanations
src/parcel/monitoring.py    Input drift and post-outcome checks
src/parcel/cli.py           Command-line interface
src/parcel/api.py           FastAPI service
src/parcel/web/             Operations dashboard
scripts/build_report.py     Generates docs/REPORT.pdf and docs/EVALUATION.md
tests/                      Unit, integration, and API tests
docs/                       Memo, plan, check-ins, walkthrough, report, AI disclosure
FAILURES.md                 Actual failure/recovery log
HOURS.md                    Human time log template (not fabricated)
```

## Honest limitations

- All results are synthetic. The generator plants the flaws the audit finds; a real extract may contain different or additional issues.
- The selected model has modest discrimination and **low recall at the economic threshold**. See `docs/REPORT.pdf` and `docs/DECISION_MEMO.md`; the recommendation is a limited shadow pilot, not a broad launch.
- The new carrier CAR_D has no pre-update training history and is a measured weak segment.
- Intervention efficacy and avoided late cost are assumptions, not measured causal effects.
- Explanations are local sensitivities, not causal or additive attributions.
- The temporal split purges immature labels, but overlapping promises and shared carrier/warehouse conditions can still leak context. A production point-in-time backtest is required.
- No authentication, rate limiting, persistent queue, or carrier write-back is implemented. Do not expose this demo publicly with real data.

## Submission documents

- Report: `docs/REPORT.pdf` (generated, ≤4 pages) and `docs/EVALUATION.md`
- Operations memo: `docs/DECISION_MEMO.md`
- Failure log: `FAILURES.md`
- Draft daily check-ins: `docs/checkins/day-1.md`, `day-2.md`, `day-3.md`
- Walkthrough script: `docs/WALKTHROUGH.md`
- Plan and risks: `docs/PLAN.md`
- AI assistance disclosure: `docs/AI_USE.md`
- Human hours: `HOURS.md` (candidate must fill actual hours)

The check-ins are **drafts**, not claims that messages were sent. The walkthrough is a script for a live or recorded 5–8 minute review; no recording is fabricated.
