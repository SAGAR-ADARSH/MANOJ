# 5–8 minute walkthrough script

Use this for a live screen-share or record it. Do not claim a recording exists unless you actually make one.

## 0:00–0:45 — Frame the problem

“I chose Project Parcel because I wanted to build a decision system, not just a classifier. The question is: at 06:00, before dispatch, which parcels are likely to miss the original customer promise, and which interventions are worth their cost under hard capacity limits? Everything here is synthetic and labeled as such.”

## 0:45–2:00 — Show the audit and leakage boundary

Open the **Data audit** tab.

“Here are the planted flaws: duplicates, missing values, impossible weights, pound/kilogram mixing, carrier aliases, and mixed timezone offsets. Each has an explicit decision. Conflicting identities are quarantined, not guessed. The label is delivery after the original promise. Delivery time, final status, rating, actual cost, and revised promises are never features.”

Show `docs/PLAN.md` or `artifacts/DATA_AUDIT.md` briefly.

## 2:00–3:30 — Show the evaluation honestly

Open **Model trust**.

“I compared a carrier-history baseline, logistic regression, and gradient boosting. The split is chronological: fit, calibrate, select, then a forward update test. I purge rows whose outcomes would not be mature by the next boundary. Logistic regression won on selection-window expected net benefit. On the forward test, AUC is 0.683 versus 0.626 for the baseline, precision at capacity is about 63%, but recall is only 2.8%. The new carrier is a clear weak spot. I am not hiding that.”

## 3:30–5:00 — Show the decision queue and explanation

Open **Action queue**, select a flagged parcel.

“The system ranks by expected net benefit: probability times assumed success times avoided late cost, minus carrier intervention cost. It enforces 80 per day, 25 per carrier, and warehouse caps. This parcel’s reason is a local sensitivity: replacing one input with a typical training value lowers estimated risk. It is a clue, not a causal proof.”

Upload `artifacts/demo_input.csv` to show validation and scoring. Then show a bad CSV being rejected.

## 5:00–6:15 — Show failure and recovery

Open `FAILURES.md`.

“The demo failed three times: historical missing values leaked into the serving demo, pandas string dtypes broke the contract join, and monitoring received raw timestamp strings. I fixed the boundaries rather than weakening validation, and added regression tests. The first test run then caught two more API/optional-column bugs. All 26 tests now pass.”

## 6:15–7:30 — Recommendation and next steps

Open `docs/DECISION_MEMO.md`.

“My recommendation is a two-week shadow pilot, not automatic dispatch changes. The benefit depends on unmeasured intervention efficacy. Next I would replace synthetic data with the official extract, run a point-in-time backtest, measure real intervention cost and success, and keep human approval before customer contact. The system is usable and explainable, but the honest business result is modest.”

## Optional close

“The main thing I want you to evaluate is not the AUC. It is whether the label, split, cost assumptions, capacity limits, and failure behavior are visible enough that you can disagree with me precisely.”
