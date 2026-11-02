# Decision memo for the operations head

**Subject:** Should we use the parcel delay desk to decide which deliveries to intervene on?

**Recommendation:** Run a **two-week shadow pilot only**. Do not let the system automatically reroute, upgrade, or message customers yet.

## What the system does

Every morning at 06:00, it looks at parcels that have not yet been dispatched. It estimates how likely each one is to arrive after the promise we originally made to the customer. It then recommends an intervention only when the expected saving is larger than the cost, and only within the team’s daily limits.

It does not flag everything. In the current demonstration, the money limit is stricter than the people limit: the team could handle 80 interventions a day, but the system usually finds only a small number worth paying for.

## What we measured

This was tested on **generated demonstration data**, not our real shipments. On the final forward test window:

- About **30%** of parcels were late.
- The system’s ranking was better than a simple carrier-history rule: ROC AUC **0.683 vs 0.626**.
- Of the parcels it recommended for intervention, about **63%** were actually late.
- But it recommended intervention for only about **2.8%** of all late parcels.
- Estimated net benefit was about **₹23 per day** under the assumed saving and success rate.

In plain terms: when it raises its hand, it is often right, but it raises its hand rarely, and the estimated money benefit is small.

## What it could cost

The demonstration assumes an intervention costs ₹60–₹110 depending on carrier, a missed late delivery costs ₹450, and an intervention prevents lateness 40% of the time. Those numbers are placeholders. If the true success rate is lower, or the true cost is higher, the benefit can disappear.

## What could go wrong

- A new carrier has no history, and the system did not flag its late parcels in the test.
- The system can be overconfident when carrier behavior changes.
- The “reason” shown for a parcel is a clue, not proof of cause.
- The demonstration data was designed to contain known problems. Real data may break it in new ways.
- The daily limits are enforced, but the ranking method is a simple greedy rule, not a guaranteed best possible assignment.

## What to do next

1. Run the system beside the current process for two weeks. Compare its recommendations with what the team actually did.
2. Measure the real cost and success rate of interventions on a small, approved set of parcels.
3. Check results separately by carrier and warehouse, especially any new carrier.
4. Keep a human approval step before any customer contact or carrier change.
5. Turn on a daily alarm for unusual input mix and a weekly check of accuracy after delivery outcomes arrive.

## What I am unsure about

I do not yet know whether the intervention benefit is real, whether the original promise is the right business target, or whether the model will survive a real carrier disruption. The honest next step is a measured pilot, not a broad launch.
