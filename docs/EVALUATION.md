# Evaluation notes — synthetic demonstration

> All numbers below come from generated data and a simulated forward update. They are not real-company performance.

## Split

- Train: 2025-01-06 00:30:00+00:00 to 2025-05-05 00:30:00+00:00 (n=12,854)
- Calibration: 2025-05-06 00:30:00+00:00 to 2025-05-20 00:30:00+00:00 (n=1,335)
- Selection: 2025-05-21 00:30:00+00:00 to 2025-06-04 00:30:00+00:00 (n=1,331)
- Forward test: 2025-06-05 00:30:00+00:00 to 2025-07-04 00:30:00+00:00 (n=4,281)
- Label-maturity purge: {'train': 296, 'calibration': 306, 'selection': 293}

## Model comparison

Selected model: **logistic**. Rule: Highest selection-window expected net benefit under hard capacities; Brier tie-break. Test not used to choose.

| Model | Selection expected net benefit | Selection ROC AUC | Selection Brier |
|---|---:|---:|---:|
| logistic | Rs. 154 | 0.712 | 0.144 |
| hist_gradient_boosting | Rs. 73 | 0.696 | 0.146 |

## Forward test: selected model vs same-window baseline

| Metric | Selected model | Historical-rate baseline |
|---|---:|---:|
| n | 4,281 | 4,281 |
| Late rate | 30.0% | 30.0% |
| ROC AUC | 0.683 | 0.626 |
| PR AUC | 0.471 | 0.385 |
| Brier | 0.202 | 0.204 |
| Flagged | 57 | 0 |
| Precision at capacity | 63.2% | 0.0% |
| Recall at capacity | 2.8% | 0.0% |
| Expected net benefit | Rs. 701 | Rs. 0 |
| Expected benefit/day | Rs. 23 | Rs. 0 |

Day-1 pre-update anchor (different window, not head-to-head): recall 0.0%, ROC AUC 0.640.

## Uncertainty

- Method: 400 whole-day bootstrap resamples; synthetic data only
- Recall at capacity 95% whole-day bootstrap interval: 1.9% – 3.8%
- Expected benefit/day 95% interval: Rs. 15 – Rs. 32

## Weakest segments (n ≥ 30)

| Segment | Value | n | Late rate | Recall at capacity | Precision | Brier |
|---|---|---:|---:|---:|---:|---:|
| carrier_id | CAR_A | 1,251 | 14.1% | 0.0% | 0.0% | 0.118 |
| carrier_id | CAR_D | 1,098 | 43.9% | 0.0% | 0.0% | 0.287 |
| service_level | express | 932 | 21.0% | 0.0% | 0.0% | 0.165 |
| destination_region | SOUTH | 863 | 28.5% | 0.4% | 16.7% | 0.200 |
| parcel_type | fragile | 596 | 29.0% | 1.2% | 40.0% | 0.201 |
| destination_region | CENTRAL | 671 | 28.9% | 2.1% | 100.0% | 0.194 |
| parcel_type | standard | 3,413 | 29.4% | 2.1% | 63.6% | 0.200 |
| destination_region | WEST | 813 | 31.9% | 2.3% | 75.0% | 0.214 |
| warehouse_id | WH_NORTH | 1,773 | 30.3% | 2.4% | 56.5% | 0.202 |
| warehouse_id | WH_SOUTH | 1,459 | 30.6% | 2.7% | 66.7% | 0.205 |

## Global importance (selection-window permutation, mean ROC AUC drop)

| Feature | Mean drop | Std |
|---|---:|---:|
| carrier_id | 0.116 | 0.023 |
| distance_km | 0.054 | 0.003 |
| service_level | 0.030 | 0.002 |
| warehouse_utilization | 0.009 | 0.002 |
| parcel_type | 0.007 | 0.001 |
| is_weekend | 0.006 | 0.003 |
| destination_region | 0.004 | 0.003 |
| weight_kg | 0.003 | 0.001 |

## Audit and drift

- Input rows: 20,854; clean: 20,696; quarantined: 55.
- Demo-batch drift status: **warn**.
- Interpretation: input drift is a warning, not proof of accuracy or failure. Post-outcome checks are still required.

## What these numbers do not show

- Real-world performance, causal intervention benefit, or production readiness.
- Optimality of greedy allocation under overlapping constraints.
- Robustness to unseen real data flaws beyond the planted synthetic issues.
