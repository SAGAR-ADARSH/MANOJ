# Simulated updates and how the system reacted

No official interviewer updates were supplied. These are explicit simulations so the full three-day structure can be demonstrated without pretending the surprises were real.

## Simulated Day 2 — data update

**Change:** 30 forward days, higher volume, new carrier CAR_D, pound-unit weights, carrier aliases, mixed timezone offsets, and CAR_C service deterioration.

**Reaction:**

1. Re-audited from raw data rather than patching the old clean file.
2. Normalized lb → kg and carrier aliases through the reference table.
3. Parsed explicit timezone offsets to UTC and derived the business day in IST.
4. Kept the update as a forward test window instead of randomly mixing it into training.
5. Surfaced CAR_D as an unseen-category drift signal and a measured weak segment.

Evidence: `artifacts/update_diff.json`, `artifacts/DATA_AUDIT.md`, `docs/EVALUATION.md`.

## Simulated Day 3 — operational constraint

**Change:** hard capacity of 80 interventions/day total, 25/day/carrier, plus warehouse caps 35/30/25.

**Reaction:**

1. Added a positive-expected-benefit gate and deterministic greedy allocation under all three caps.
2. Added CLI/API/dashboard paths that expose the allocation reason and capacity rank.
3. Added strict serving validation, local explanations, global importance, and drift monitoring.
4. Wrote a shadow-pilot decision memo instead of claiming the constraint proves readiness.

**Honest sequencing caveat:** for a compact, reproducible prototype, the final policy is applied consistently during selection and evaluation so model comparisons use the same decision rule. This is not a true sequential “surprise after Day 2” experiment. When the official Day 3 constraint arrives, re-run `parcel train --policy <official-policy.json>` and compare against the frozen prior artifacts before changing the recommendation.
