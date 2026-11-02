# Plan and top risks

## Framing

Predict whether a parcel will arrive after its **original customer promise**, using only information available at the **06:00 Asia/Kolkata dispatch snapshot**. Recommend an intervention only when its assumed expected benefit exceeds its cost and hard operational capacity allows it.

## Day 1 — understand and anchor

1. Define the label and prediction cutoff; write the column availability table.
2. Generate a deterministic synthetic extract with planted flaws: duplicates, missingness, impossible weights, mixed units, aliases, mixed timezone offsets, and post-outcome fields.
3. Build audit/cleaning with explicit quarantine.
4. Fit a smoothed carrier×service historical-rate baseline and evaluate on a forward pre-update window.
5. Write the plan and risks.

## Day 2 — survive an update

1. Simulate a forward update: new carrier, higher volume, pound units, aliases, mixed offsets, and CAR_C deterioration.
2. Re-audit from raw data; normalize units/aliases; keep the update as a forward test window.
3. Compare logistic regression and histogram gradient boosting; calibrate on a dedicated chronological window; select on a separate window; test once.
4. Evaluate with ROC AUC, PR AUC, Brier, precision/recall at capacity, and expected net benefit.
5. Break the model by carrier, region, warehouse, service, and parcel type.

## Day 3 — make it usable and trustworthy

1. Simulate an operational constraint: 80/day total, 25/day/carrier, warehouse caps.
2. Package CLI + API + dashboard; validate inputs strictly.
3. Add local explanations, global permutation importance, and input-drift monitoring.
4. Add tests for leakage boundaries, allocation caps, validation, monitoring, and API behavior.
5. Write the decision memo, report, walkthrough, and failure log.

## Top three risks

| Risk | Why it matters | Mitigation in this prototype | Remaining gap |
|---|---|---|---|
| Leakage / optimistic evaluation | Post-outcome fields or immature labels can fake performance | Explicit feature allowlist; chronological split; label-maturity purge; untouched forward test | Real point-in-time backtest still required |
| Distribution shift / new carrier | Day 2 changes carrier mix and service quality | Forward update test; unseen-category drift alarm; segment report | CAR_D remains a measured weak spot |
| False economic confidence | Expected benefit depends on unmeasured efficacy and cost | Explicit assumptions; positive-benefit gate; hard caps; shadow-pilot recommendation | No causal intervention experiment |

## What was deliberately not built

- A deep model or AutoML search: the assignment rewards defensible decisions, not complexity.
- Carrier write-back or automatic customer messaging: unsafe without auth, approval, and audit trails.
- A production database/queue: CSV + artifacts are enough to demonstrate the decision path honestly.
- Fabricated human hours or sent check-ins: the candidate must supply those.
