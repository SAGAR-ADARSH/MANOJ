# Failure and Recovery Log

This log records events that actually occurred during this implementation. It is not a retrospective reconstruction of three human working days.

## F-001 — System Python refused the installation

- **Context:** First dependency installation, before implementation.
- **What happened:** `python -m pip install -e '.[dev]'` failed with `externally-managed-environment` (PEP 668).
- **What I believed:** The sandbox would allow installing into the default Python environment.
- **What I tried:** Created a project-local `.venv` and installed the pinned dependencies there instead of overriding the OS package manager.
- **What I learned:** The README must start with virtual-environment creation. A clean checkout is not the same as a clean Python environment.
- **Status:** Recovery implemented; installation result recorded in the build log.

## F-002 — The first end-to-end demo failed at inference

- **Context:** First `parcel demo` run after training completed.
- **What happened:** Training and evaluation succeeded, but scoring the exported demo batch failed with `weight_kg: missing or non-numeric value`.
- **What I believed:** A historical test row could be reused unchanged as a live inference example.
- **What I tried:** Inspected the exported batch: it contained historical rows with missing weight/distance. The training path intentionally tolerates and imputes those; the live path intentionally rejects them.
- **What I learned:** Training and serving contracts were correctly different, but the demo exporter had not respected the serving contract. The fix belongs in the exporter, not in weakening live validation.
- **Fix:** Export a complete-case demo batch and document that historical missingness is not silently imputed at serving time.
- **Status:** Fixed; regression test added for complete-case demo export and strict inference rejection.

## F-003 — Serving failed on a pandas dtype mismatch

- **Context:** Second end-to-end run, after fixing the demo exporter.
- **What happened:** `parcel predict` failed inside `merge_asof` with `incompatible merge keys [0] string[python] and dtype('O')`.
- **What I believed:** Normalizing carrier IDs to pandas nullable strings in validation would be transparent to the contract join.
- **What I tried:** Reproduced the serving path and compared it with training. Training used object-dtype IDs from CSV; serving used `string[python]` after validation.
- **What I learned:** Semantic equality is not enough for pandas merge keys. The serving path needs an explicit dtype boundary, and tests must exercise validation → contract join → prediction, not just the model in isolation.
- **Fix:** Cast both sides of the effective-dated contract join to plain `str` before `merge_asof`; add an integration test through the public prediction function.
- **Status:** Fixed; verified by the end-to-end demo and tests.

## F-004 — Drift monitoring received raw CSV strings

- **Context:** End-to-end demo after prediction was fixed.
- **What happened:** `drift_check` crashed subtracting timestamp strings because the demo command passed a freshly read CSV rather than validated, parsed rows.
- **What I believed:** Feature engineering could accept either raw CSV frames or validated frames.
- **What I tried:** Traced the call path: `predict` validates internally, but the separate monitoring call bypassed validation.
- **What I learned:** Shared feature engineering needs one explicit input boundary. Monitoring must not silently accept a different contract from serving.
- **Fix:** Validate before monitoring in the CLI; in the API, monitor the already validated/scored frame. Add a regression test that runs drift checks on the demo batch.
- **Status:** Fixed; verified end to end.

## F-005 — First test run exposed two boundary bugs

- **Context:** First full `pytest` run after the CLI demo passed.
- **What happened:** 3 of 26 tests failed. (1) The parcel-detail API passed CSV-loaded timestamp strings into feature engineering. (2) The audit assumed a `delivered_on_time` column even when a minimal historical extract omitted it.
- **What I believed:** The CLI happy path covered the same boundaries as the API and minimal extracts.
- **What I tried:** Read the tracebacks rather than relaxing assertions. The API needed the same timestamp normalization as serving; the audit needed an explicit column-presence guard.
- **What I learned:** A passing demo is not a passing contract. API serialization and optional-column handling need their own tests.
- **Fix:** Parse all timestamp columns when loading scored test data; guard the optional on-time-flag comparison. Re-ran the suite.
- **Status:** Fixed; tests pass.

## Scope corrections (not runtime failures)

- The repository contained no company assets. The user explicitly chose a **synthetic demo** and a **full three-day prototype**. No synthetic result is presented as company performance.
- No official Day 2 or Day 3 update was supplied. Both updates are explicitly simulated.
- No human time log or daily check-in transmission is invented. Check-in files are draft milestone notes for the candidate to review and send.
