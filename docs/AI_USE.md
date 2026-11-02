# AI assistance disclosure

This project was implemented in Arena.ai Agent Mode with substantial coding-agent assistance.

## Where assistance was used

- Repository inspection and project scaffolding.
- Python implementation: synthetic data generation, audit/cleaning, schema validation, modeling, allocation, explanations, monitoring, CLI, API, and dashboard.
- Test authoring and debugging from actual tracebacks.
- Documentation drafts: README, plan, memo, check-in templates, walkthrough script, and report generation.

## Where human judgment is still required

- The candidate must review and be able to explain every line.
- The candidate must record actual hours in `HOURS.md`.
- The candidate must review, personalize, and send the daily check-ins.
- The candidate must deliver the live or recorded walkthrough and defend decisions in the 30-minute review.
- The candidate must replace synthetic assets and simulated updates with official company assets/updates when provided.

## Verification performed by the agent

- `pip install -e ".[dev]"` in a project-local virtual environment.
- `parcel demo` end to end.
- `pytest` (26 tests) and `ruff check src tests`.
- API smoke checks through FastAPI TestClient.

No external hosted language model is used at runtime. The runtime is deterministic Python/scikit-learn/FastAPI.
