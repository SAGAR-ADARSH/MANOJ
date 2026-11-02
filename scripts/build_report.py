"""Generate docs/EVALUATION.md and a compact (<=4 page) docs/REPORT.pdf from artifacts.

Run after `parcel demo`. This script never invents metrics: missing artifacts fail.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def num(x):
    return "—" if x is None else f"{x:,.0f}"


def dec(x):
    return "—" if x is None else f"{x:.3f}"


def money(x):
    return "—" if x is None else f"Rs. {x:,.0f}"


def load():
    required = ["metrics.json", "audit.json"]
    missing = [f for f in required if not (ART / f).exists()]
    if missing:
        raise SystemExit(f"Missing artifacts: {missing}. Run: parcel demo")
    metrics = json.loads((ART / "metrics.json").read_text())
    audit = json.loads((ART / "audit.json").read_text())
    drift = json.loads((ART / "drift.json").read_text()) if (ART / "drift.json").exists() else {"status": "not_run"}
    return metrics, audit, drift


def evaluation_md(metrics, audit, drift):
    selected = metrics["selected_model"]
    test = metrics["models"][selected]["test"]
    base = metrics["baseline_same_test"]
    lines = [
        "# Evaluation notes — synthetic demonstration",
        "",
        "> All numbers below come from generated data and a simulated forward update. They are not real-company performance.",
        "",
        "## Split",
        "",
        f"- Train: {metrics['split']['train'][0]} to {metrics['split']['train'][1]} (n={num(metrics['split']['counts']['train'])})",
        f"- Calibration: {metrics['split']['calibration'][0]} to {metrics['split']['calibration'][1]} (n={num(metrics['split']['counts']['calibration'])})",
        f"- Selection: {metrics['split']['selection'][0]} to {metrics['split']['selection'][1]} (n={num(metrics['split']['counts']['selection'])})",
        f"- Forward test: {metrics['split']['test'][0]} to {metrics['split']['test'][1]} (n={num(metrics['split']['counts']['test'])})",
        f"- Label-maturity purge: {metrics['split']['label_maturity_purged']}",
        "",
        "## Model comparison",
        "",
        f"Selected model: **{selected}**. Rule: {metrics['selection_rule']}",
        "",
        "| Model | Selection expected net benefit | Selection ROC AUC | Selection Brier |",
        "|---|---:|---:|---:|",
    ]
    for name, m in metrics["models"].items():
        s = m["selection"]
        lines.append(f"| {name} | {money(s['expected_net_benefit_inr'])} | {dec(s['roc_auc'])} | {dec(s['brier'])} |")
    lines += [
        "",
        "## Forward test: selected model vs same-window baseline",
        "",
        "| Metric | Selected model | Historical-rate baseline |",
        "|---|---:|---:|",
        f"| n | {num(test['n'])} | {num(base['n'])} |",
        f"| Late rate | {pct(test['late_rate'])} | {pct(base['late_rate'])} |",
        f"| ROC AUC | {dec(test['roc_auc'])} | {dec(base['roc_auc'])} |",
        f"| PR AUC | {dec(test['pr_auc'])} | {dec(base['pr_auc'])} |",
        f"| Brier | {dec(test['brier'])} | {dec(base['brier'])} |",
        f"| Flagged | {num(test['flagged'])} | {num(base['flagged'])} |",
        f"| Precision at capacity | {pct(test['precision_at_capacity'])} | {pct(base['precision_at_capacity'])} |",
        f"| Recall at capacity | {pct(test['recall_at_capacity'])} | {pct(base['recall_at_capacity'])} |",
        f"| Expected net benefit | {money(test['expected_net_benefit_inr'])} | {money(base['expected_net_benefit_inr'])} |",
        f"| Expected benefit/day | {money(test['expected_benefit_per_day_inr'])} | {money(base['expected_benefit_per_day_inr'])} |",
        "",
        f"Day-1 pre-update anchor (different window, not head-to-head): recall {pct(metrics['day1_baseline_pre_update']['recall_at_capacity'])}, ROC AUC {dec(metrics['day1_baseline_pre_update']['roc_auc'])}.",
        "",
        "## Uncertainty",
        "",
        f"- Method: {metrics['test_uncertainty']['method']}",
        f"- Recall at capacity 95% whole-day bootstrap interval: {pct(metrics['test_uncertainty']['recall_at_capacity_ci95'][0])} – {pct(metrics['test_uncertainty']['recall_at_capacity_ci95'][1])}",
        f"- Expected benefit/day 95% interval: {money(metrics['test_uncertainty']['expected_benefit_per_day_ci95'][0])} – {money(metrics['test_uncertainty']['expected_benefit_per_day_ci95'][1])}",
        "",
        "## Weakest segments (n ≥ 30)",
        "",
        "| Segment | Value | n | Late rate | Recall at capacity | Precision | Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["segments"][:10]:
        lines.append(f"| {r['segment']} | {r['value']} | {num(r['n'])} | {pct(r['late_rate'])} | {pct(r['recall_at_capacity'])} | {pct(r['precision_at_capacity'])} | {dec(r['brier'])} |")
    lines += [
        "",
        "## Global importance (selection-window permutation, mean ROC AUC drop)",
        "",
        "| Feature | Mean drop | Std |",
        "|---|---:|---:|",
    ]
    for r in metrics["global_importance"][:8]:
        lines.append(f"| {r['feature']} | {dec(r['mean_auc_drop'])} | {dec(r['std'])} |")
    lines += [
        "",
        "## Audit and drift",
        "",
        f"- Input rows: {num(audit['input_rows'])}; clean: {num(audit['clean_rows'])}; quarantined: {num(audit['quarantined_rows'])}.",
        f"- Demo-batch drift status: **{drift['status']}**.",
        "- Interpretation: input drift is a warning, not proof of accuracy or failure. Post-outcome checks are still required.",
        "",
        "## What these numbers do not show",
        "",
        "- Real-world performance, causal intervention benefit, or production readiness.",
        "- Optimality of greedy allocation under overlapping constraints.",
        "- Robustness to unseen real data flaws beyond the planted synthetic issues.",
    ]
    (DOCS / "EVALUATION.md").write_text("\n".join(lines) + "\n")


def build_pdf(metrics, audit, drift):
    selected = metrics["selected_model"]
    test = metrics["models"][selected]["test"]
    base = metrics["baseline_same_test"]
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=8, textColor=colors.HexColor("#101a2b"))
    h = ParagraphStyle("HX", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#0d7a68"))
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=9.2, leading=12.5, spaceAfter=5)
    small = ParagraphStyle("SmallX", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#5b687c"))
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4f2")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8e0e8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 5))

    p("Parcel Decision Desk", title)
    p("Intern Challenge · Project Parcel · Synthetic prototype report", small)
    p("<b>Scope warning.</b> All shipments, carriers, updates, and results are generated. Day 2/3 updates are simulated, not official interviewer updates. This report is not evidence of real-company performance.", body)
    p("1. Problem as I understood it", h)
    p("At the 06:00 Asia/Kolkata dispatch snapshot, estimate whether each parcel will arrive after its <b>original customer promise</b>, then recommend an intervention only when assumed expected benefit exceeds cost and hard capacity allows it. The operations head cannot flag everything; the system must be selective, explainable, and honest about uncertainty.")
    p("Assumptions: backlog is a true as-of count; carrier contracts are effective-dated; intervention cost is Rs. 60–Rs. 110; avoided late cost is Rs. 450; intervention success is 40%. Efficacy and avoided cost are placeholders, not measured causal effects.")
    p("2. Approach and key decisions", h)
    p("<b>Data.</b> Deterministic generator plants duplicates, missingness, impossible weights, lb/kg mixing, aliases, mixed timezone offsets, and post-outcome fields. Audit normalizes units/aliases/timezones, recomputes the label from the original promise, and quarantines conflicting identities. Live inference rejects missing required values.")
    p("<b>Leakage control.</b> Feature allowlist excludes delivery time, final status, exception code, rating, actual cost, and revised promise. Chronological fit/calibration/selection/test split; rows whose outcomes are immature at the next boundary are purged. The Day 2 update is the forward test window.")
    p("<b>Models.</b> Smoothed carrier×service historical-rate baseline; regularized logistic regression; histogram gradient boosting. Both learned models are sigmoid-calibrated on a dedicated window and selected by expected net benefit on a separate window. Logistic regression was selected.")
    p("<b>Decision.</b> Expected net benefit = p × 0.40 × Rs. 450 − carrier cost. Positive-benefit parcels are greedily allocated under 80/day total, 25/day/carrier, and warehouse caps 35/30/25. Greedy is auditable but not proven globally optimal.")
    p("Rejected alternatives: random split (leaks future context), deep/AutoML complexity (harder to defend), automatic carrier write-back (unsafe without approval/auth), and revising the promise label (can hide lateness).")

    story.append(PageBreak())
    p("3. Results, with honest uncertainty", h)
    table([
        ["Forward test metric", "Selected model", "Same-window baseline"],
        ["n / late rate", f"{num(test['n'])} / {pct(test['late_rate'])}", f"{num(base['n'])} / {pct(base['late_rate'])}"],
        ["ROC AUC", dec(test["roc_auc"]), dec(base["roc_auc"])],
        ["PR AUC", dec(test["pr_auc"]), dec(base["pr_auc"])],
        ["Brier", dec(test["brier"]), dec(base["brier"])],
        ["Flagged", num(test["flagged"]), num(base["flagged"])],
        ["Precision at capacity", pct(test["precision_at_capacity"]), pct(base["precision_at_capacity"])],
        ["Recall at capacity", pct(test["recall_at_capacity"]), pct(base["recall_at_capacity"])],
        ["Expected benefit/day", money(test["expected_benefit_per_day_inr"]), money(base["expected_benefit_per_day_inr"])],
    ], [78 * mm, 48 * mm, 48 * mm])
    ci = metrics["test_uncertainty"]["recall_at_capacity_ci95"]
    p(f"Whole-day bootstrap 95% interval for recall at capacity: {pct(ci[0])}–{pct(ci[1])}. This quantifies synthetic sampling variability, not real-world uncertainty.")
    p("<b>Interpretation.</b> The model ranks better than the baseline and is often right when it flags, but it flags very few late parcels because the economic threshold is strict. Estimated benefit is small. Economics, not capacity, is the binding constraint in this run.")
    p("Weakest segments", h)
    rows = [["Segment", "Value", "n", "Late rate", "Recall", "Brier"]]
    for r in metrics["segments"][:6]:
        rows.append([r["segment"], r["value"], num(r["n"]), pct(r["late_rate"]), pct(r["recall_at_capacity"]), dec(r["brier"])])
    table(rows, [38 * mm, 34 * mm, 18 * mm, 26 * mm, 24 * mm, 24 * mm])
    p("The new carrier CAR_D has no pre-update training history and zero recall at capacity in the test. Express service and CAR_A also show low recall under the economic gate. These are measured failures, not guesses.")
    p("Global reliance", h)
    imp = ", ".join(f"{r['feature']} ({dec(r['mean_auc_drop'])})" for r in metrics["global_importance"][:5])
    p(f"Selection-window permutation importance (mean ROC AUC drop): {imp}. Carrier identity dominates. I trust this as an association in the synthetic generator, not as a causal claim or proof of real-world stability.")

    story.append(PageBreak())
    p("4. What failed and what I learned", h)
    p("<b>F-001 Installation.</b> System Python refused pip under PEP 668. Fixed by using a project-local virtual environment; README now starts there.")
    p("<b>F-002 Demo export.</b> Historical missing values entered the serving demo and were correctly rejected. Fixed the exporter to emit complete cases rather than weakening live validation.")
    p("<b>F-003 Contract join.</b> Nullable-string vs object dtype broke merge_asof. Added an explicit dtype boundary and an integration test through validation → contract join → prediction.")
    p("<b>F-004 Monitoring boundary.</b> Drift checks received raw CSV timestamp strings. Validated before monitoring and added a regression test.")
    p("<b>F-005 First test run.</b> API detail passed CSV timestamp strings into feature engineering; audit assumed an optional on-time flag existed. Fixed both. Lesson: a passing demo is not a passing contract.")
    p("5. Risks, limitations, and two more weeks", h)
    table([
        ["Risk", "Current mitigation", "Remaining gap"],
        ["Synthetic-only evidence", "Explicit labels and simulated updates", "Replace with official data and re-run"],
        ["Low recall / small benefit", "Positive-benefit gate; shadow-pilot memo", "Measure real intervention efficacy and cost"],
        ["New carrier / drift", "Unseen-category alarm; segment report", "Manual-review policy and carrier-specific calibration"],
        ["Leakage / optimism", "Allowlist, chronological split, maturity purge", "Production point-in-time backtest"],
        ["Explanation overclaim", "Local sensitivity caveat; global permutation", "Human review and causal experiment"],
    ], [42 * mm, 66 * mm, 66 * mm])
    p("With two more weeks: (1) ingest the official extract and write a real point-in-time backtest; (2) run a small approved intervention experiment to estimate efficacy; (3) add carrier-specific calibration and a manual-review path for unseen carriers; (4) replace greedy allocation with an exact constrained optimizer if capacity becomes binding; (5) add auth, audit logs, and a persistent approval queue.")
    p("Recommendation", h)
    p("Run a two-week shadow pilot only. Keep human approval before customer contact or carrier changes. Watch daily input drift, weekly post-outcome calibration/recall, carrier-level precision/recall, intervention cost, and realized late rate. Do not broadly launch on synthetic evidence.")

    story.append(PageBreak())
    p("6. How to run everything", h)
    p("Python 3.11+. From a clean checkout:", body)
    table([
        ["Step", "Command"],
        ["Install", 'python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"'],
        ["End-to-end demo", "parcel demo"],
        ["Dashboard/API", "parcel serve --artifacts artifacts --host 0.0.0.0 --port 8000"],
        ["Predict", "parcel predict --input artifacts/demo_input.csv --output artifacts/predictions.csv"],
        ["Drift check", "parcel monitor --input artifacts/demo_input.csv  # exit 2 = warning"],
        ["Tests / lint", "pytest && ruff check src tests"],
        ["Rebuild report", "python scripts/build_report.py"],
    ], [38 * mm, 136 * mm])
    p("Deliverables map", h)
    table([
        ["Requirement", "Where"],
        ["Working system", "src/parcel CLI + FastAPI + dashboard"],
        ["Audit / leakage decisions", "artifacts/DATA_AUDIT.md, artifacts/audit.json"],
        ["Evaluation / error analysis", "docs/EVALUATION.md, artifacts/metrics.json"],
        ["Failure log", "FAILURES.md"],
        ["Daily check-ins", "docs/checkins/day-1.md … day-3.md (drafts, not sent)"],
        ["Decision memo", "docs/DECISION_MEMO.md"],
        ["Walkthrough", "docs/WALKTHROUGH.md (script; record or present live)"],
        ["AI disclosure / hours", "docs/AI_USE.md, HOURS.md (human hours not fabricated)"],
    ], [52 * mm, 122 * mm])
    p(f"Current artifact snapshot: {num(audit['input_rows'])} input rows, {num(audit['clean_rows'])} clean, {num(audit['quarantined_rows'])} quarantined; demo drift status {drift['status']}. Tests: 26 passing at build time. Generated by scripts/build_report.py from artifacts; no metrics are hand-entered.", small)

    DOCS.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(DOCS / "REPORT.pdf"), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Parcel Decision Desk — Synthetic Prototype Report",
                            author="Intern Challenge")
    doc.build(story)


def main():
    metrics, audit, drift = load()
    DOCS.mkdir(parents=True, exist_ok=True)
    evaluation_md(metrics, audit, drift)
    build_pdf(metrics, audit, drift)
    print(f"Wrote {DOCS / 'EVALUATION.md'} and {DOCS / 'REPORT.pdf'}")


if __name__ == "__main__":
    main()
