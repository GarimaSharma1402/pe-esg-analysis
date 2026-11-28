"""Assemble paper tables from recorded user and assistant screening reviews."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from .annotation import reviewed_annotations
from .io import load_companies, write_json
from .paper_results import _evaluation_figure
from .schemas import RELEVANCE_LEVELS, RISK_CATEGORIES


OUTCOMES = (
    "user_climate_positive", "ai_climate_candidate", "ai_needs_context",
    "user_not_climate", "ai_not_climate_in_record",
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_reviews(
    predictions: pd.DataFrame, decisions: pd.DataFrame, source_checks: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    for name, frame in (("predictions", predictions), ("decisions", decisions), ("source checks", source_checks)):
        if frame["record_id"].eq("").any() or frame["record_id"].duplicated().any():
            raise ValueError(f"{name} must have unique non-empty record IDs")
    reviewed = reviewed_annotations(annotations)
    if not set(decisions["review_outcome"]).issubset(OUTCOMES):
        raise ValueError("Unknown review outcome")
    if not decisions["model_climate_positive"].isin(("True", "False")).all():
        raise ValueError("Invalid model climate flag")
    user_positive = reviewed[reviewed["is_climate_related"].str.strip().str.lower().isin(("yes", "true", "1"))]
    required = set(predictions.loc[predictions["is_climate_related"].eq("True"), "record_id"]) | set(user_positive["record_id"])
    if set(decisions["record_id"]) != required or set(source_checks["record_id"]) != required:
        raise ValueError("Review and source checks must cover all model flags and user-reviewed positives")
    if not source_checks["pdf_block_match"].eq("True").all() or not source_checks["manifest_match"].eq("True").all():
        raise ValueError("Source checks contain a mismatch")
    checked = decisions.merge(source_checks, on="record_id", validate="one_to_one", suffixes=("_review", "_source"))
    if not checked["company_id_review"].eq(checked["company_id_source"]).all() or not checked["pdf_review"].eq(checked["pdf_source"]).all():
        raise ValueError("Review company or PDF differs from source check")
    merged = decisions.merge(predictions, on="record_id", validate="one_to_one", suffixes=("_review", "_model"))
    if not merged["company_id_review"].eq(merged["company_id_model"]).all():
        raise ValueError("Review company differs from prediction")
    if not merged["model_climate_positive"].eq(merged["is_climate_related"]).all():
        raise ValueError("Review model flags differ from predictions")
    verified = reviewed.set_index("record_id")
    user_negative = reviewed[reviewed["is_climate_related"].str.strip().str.lower().isin(("no", "false", "0"))]
    required_negative = set(user_negative["record_id"]) & required
    if set(decisions.loc[decisions["review_outcome"].eq("user_climate_positive"), "record_id"]) != set(user_positive["record_id"]):
        raise ValueError("All user-positive labels must be retained")
    if set(decisions.loc[decisions["review_outcome"].eq("user_not_climate"), "record_id"]) != required_negative:
        raise ValueError("All user-negative labels on model flags must be retained")
    for row in decisions.itertuples(index=False):
        user_label = row.review_outcome in ("user_climate_positive", "user_not_climate")
        if user_label:
            if row.record_id not in verified.index:
                raise ValueError(f"User review not found: {row.record_id}")
            annotation = verified.loc[row.record_id]
            climate = str(annotation["is_climate_related"]).strip().lower() in ("yes", "true", "1")
            if climate != (row.review_outcome == "user_climate_positive"):
                raise ValueError(f"User review disagrees with annotation: {row.record_id}")
            if climate and (
                row.review_category != annotation["risk_category"]
                or row.user_specificity != annotation["is_specific"]
                or row.user_financial_relevance != annotation["financial_relevance"]
            ):
                raise ValueError(f"User label fields disagree with annotation: {row.record_id}")
        if row.review_outcome == "ai_climate_candidate" and row.review_category not in RISK_CATEGORIES:
            raise ValueError(f"Missing AI candidate category: {row.record_id}")
        if not user_label and (row.user_specificity or row.user_financial_relevance):
            raise ValueError(f"AI review cannot be presented as a user label: {row.record_id}")
    return merged


def _profiles(companies, predictions: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for company in companies:
        all_company = predictions[predictions["company_id"].eq(company.company_id)]
        records = reviewed[reviewed["company_id_review"].eq(company.company_id)]
        flags = records[records["model_climate_positive"].eq("True")]
        row = {
            "company_id": company.company_id, "company_name": company.company_name,
            "acquisition_announcement_date": company.acquisition_announcement_date,
            "corpus_records": len(all_company), "model_flagged_records": len(flags),
            **{name: int(flags["review_outcome"].eq(name).sum()) for name in OUTCOMES},
            "user_positive_model_missed": int(
                (records["review_outcome"].eq("user_climate_positive") & records["model_climate_positive"].eq("False")).sum()
            ),
        }
        for category in RISK_CATEGORIES:
            for basis in ("user_climate_positive", "ai_climate_candidate"):
                row[f"{basis}_{category}"] = int(
                    (records["review_outcome"].eq(basis) & records["review_category"].eq(category)).sum()
                )
        for level in RELEVANCE_LEVELS:
            row[f"user_positive_relevance_{level}"] = int(
                (records["review_outcome"].eq("user_climate_positive") & records["user_financial_relevance"].eq(level)).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _review_figure(profiles: pd.DataFrame) -> str:
    colors = ("#267a60", "#669946", "#d8a52d", "#7979a6", "#a9b0b4")
    labels = ("User positive", "AI candidate", "Needs context", "User negative", "AI not climate")
    width, x0, top, row_h = 1100, 235, 82, 25
    legend_y = top + row_h * len(profiles) + 35
    height = legend_y + 75
    scale = 680 / max(1, int(profiles["model_flagged_records"].max()))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Review of model-flagged records by company</title>',
        '<desc id="description">User annotations and AI assistant reviews shown separately. '
        'Counts are extracted records, not risk scores; model misses are not shown.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222" font-size="12">',
        '<text x="20" y="27" font-size="19" font-weight="bold">Review of model-flagged records</text>',
        '<text x="20" y="49">AI assessments are not independent human annotations; counts are not company risk scores.</text>',
    ]
    for index, row in enumerate(profiles.to_dict("records")):
        y = top + row_h * index
        parts.append(f'<text x="20" y="{y + 14}">{escape(str(row["company_name"]))}</text>')
        x = float(x0)
        for outcome, color in zip(OUTCOMES, colors, strict=True):
            w = int(row[outcome]) * scale
            if w:
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="17" fill="{color}"/>')
                x += w
        parts.append(f'<text x="{x + 5:.2f}" y="{y + 14}">{row["model_flagged_records"]}</text>')
    for i, (color, label) in enumerate(zip(colors, labels, strict=True)):
        x = 20 + i * 205
        parts.append(f'<rect x="{x}" y="{legend_y}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y + 11}">{escape(label)}</text>')
    missed = int(profiles["user_positive_model_missed"].sum())
    parts.append(f'<text x="20" y="{legend_y + 43}">{missed} user-reviewed positives missed by the model are excluded from these bars.</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def generate_reviewed_results(
    *, companies_path: str | Path, predictions_path: str | Path, decisions_path: str | Path,
    source_checks_path: str | Path, annotation_dir: str | Path, comparison_path: str | Path,
    manifest_path: str | Path, extraction_report_path: str | Path, pdf_dir: str | Path,
    output_dir: str | Path,
) -> None:
    companies_path, predictions_path, decisions_path = map(Path, (companies_path, predictions_path, decisions_path))
    source_checks_path, annotation_dir, comparison_path, output = map(
        Path, (source_checks_path, annotation_dir, comparison_path, output_dir)
    )
    manifest_path, extraction_report_path, pdf_dir = map(Path, (manifest_path, extraction_report_path, pdf_dir))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results directory: {output}")
    companies = load_companies(companies_path)
    predictions, decisions, checks = map(_read, (predictions_path, decisions_path, source_checks_path))
    annotations = pd.concat([
        _read(annotation_dir / "train_annotations.csv"),
        _read(annotation_dir / "test_annotations.csv"),
        _read(annotation_dir / "suggested_test_challenge_candidates.csv"),
    ], ignore_index=True)
    reviewed = validate_reviews(predictions, decisions, checks, annotations)
    company_by_id = {company.company_id: company for company in companies}
    if len(company_by_id) != len(companies) or set(predictions["company_id"]) != set(company_by_id):
        raise ValueError("Prediction company IDs do not match company metadata")
    if not predictions["is_before_cutoff"].eq("True").all():
        raise ValueError("Predictions include post-announcement records")
    dates = pd.to_datetime(predictions["document_date"], format="%Y-%m-%d", errors="raise").dt.date
    cutoffs = predictions["company_id"].map(lambda company_id: company_by_id[company_id].acquisition_announcement_date)
    if (dates >= cutoffs).any():
        raise ValueError("Prediction dates do not precede the announcement")
    manifest = _read(manifest_path)
    if manifest["company_id"].duplicated().any() or set(manifest["company_id"]) != set(company_by_id):
        raise ValueError("Manifest must contain one entry per company")
    manifest = manifest.set_index("company_id")
    source = reviewed.join(manifest[["filing_date", "report_url", "local_file"]], on="company_id_review")
    if not (
        source["document_date"].eq(source["filing_date"])
        & source["source"].eq(source["report_url"])
        & source["pdf"].eq(source["local_file"])
    ).all():
        raise ValueError("Reviewed record differs from manifest provenance")
    extraction = json.loads(extraction_report_path.read_text(encoding="utf-8"))
    reports = {item["company_id"]: item for item in extraction}
    if len(reports) != len(extraction) or set(reports) != set(company_by_id):
        raise ValueError("Extraction report must contain one entry per company")
    for company_id, item in reports.items():
        if item["file"] != manifest.at[company_id, "local_file"] or _hash(pdf_dir / item["file"]) != item["sha256"]:
            raise ValueError(f"PDF differs from extraction report: {company_id}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    for name, path in {
        "companies": companies_path,
        "training_disclosures": annotation_dir / "train_disclosures.csv",
        "training_annotations": annotation_dir / "train_annotations.csv",
        "random_annotations": annotation_dir / "test_annotations.csv",
        "challenge_annotations": annotation_dir / "suggested_test_challenge_candidates.csv",
    }.items():
        if comparison["input_sha256"][name] != _hash(path):
            raise ValueError(f"Comparison is stale for {name}")
    metrics = pd.DataFrame(comparison["metrics"])
    profiles = _profiles(companies, predictions, reviewed)
    software_count = sum("software" in company.industry.casefold() for company in companies)
    figure = _review_figure(profiles)
    evaluation_figure = _evaluation_figure(metrics)
    outcome_counts = decisions["review_outcome"].value_counts()
    summary = pd.DataFrame([
        {"review_outcome": outcome, "records": int(outcome_counts.get(outcome, 0))}
        for outcome in OUTCOMES
    ])
    output.mkdir(parents=True)
    profiles.to_csv(output / "company_profiles.csv", index=False)
    summary.to_csv(output / "review_outcome_summary.csv", index=False)
    metrics.to_csv(output / "climate_detection_evaluation.csv", index=False)
    reviewed.rename(columns={
        "company_id_review": "company_id", "text": "source_passage",
        "risk_category": "rule_risk_category", "financial_relevance": "rule_financial_relevance",
        "is_specific": "rule_is_specific",
    }).to_csv(output / "reviewed_evidence.csv", index=False)
    (output / "review_outcomes.svg").write_text(figure, encoding="utf-8")
    (output / "climate_detection_evaluation.svg").write_text(evaluation_figure, encoding="utf-8")
    findings = [
        "# Exploratory disclosure-screening results",
        "",
        f"ClimateBERT screened {len(predictions):,} extracted records from {len(companies)} companies and flagged "
        f"{int(predictions['is_climate_related'].eq('True').sum()):,} records. All flags were reviewed by the "
        "AI assistant or had existing user annotations. These are candidate passages, not verified risks.",
        "",
        "The review ledger includes all flags and previously user-reviewed positives missed by the model. "
        "Counts below are records, including model misses, not distinct risks.",
        "",
        "| Review outcome | Records |",
        "| --- | ---: |",
    ]
    findings.extend(f"| {outcome.replace('_', ' ')} | {int(outcome_counts.get(outcome, 0))} |" for outcome in OUTCOMES)
    findings.extend([
        "",
        "The user-reviewed positives retain the user's category, specificity and qualitative financial-relevance "
        "labels. AI candidates have assistant-assessed climate relevance and category only; their rule-based "
        "specificity and financial relevance are not validated.",
        "",
        "## Interpretation and limitations",
        "",
        f"- This is a convenience sample: {software_count} of {len(companies)} company industry labels contain "
        "'software'. Its industry composition and selection based on available reports limit generalization "
        "to the broader private-equity target population.",
        "- A disclosure signal indicates text relevant to screening, not proof of actual financial loss or "
        "a hidden risk. Public disclosures alone cannot establish that investors or ESG ratings overlooked "
        "the issue; qualitative financial-relevance labels are not measured impacts.",
        "- AI assistant review is not independent human annotation. Context-needed records remain unresolved.",
        "- Model-negative records were not comprehensively reviewed. Zero flags do not establish absence of risk.",
        "- Recorded source checks matched reviewed PDF blocks and manifest information, and PDF hashes were "
        "verified against the extraction report. These checks do not replace reading surrounding pages or "
        "confirming the publication date and completeness of each source.",
        "- Counts depend on document coverage, extraction and block splitting; they are not prevalence estimates, "
        "company risk rankings, or financial-impact estimates. Weather hazards are a screening convention, "
        "not proof of climate-change causation.",
        "- The random held-out cohort has no positive examples. The positive challenge cohort is small, selected "
        "and previously examined. The comparison table does not support reliable overall precision, F1, "
        "macro-F1 or generalization claims; do not combine these cohorts as an unbiased test.",
        "- Historical ESG-score comparison was omitted because suitable comparable, pre-announcement scores "
        "were unavailable. Do not claim an empirical comparison with aggregate ESG ratings.",
        "",
        "Before publication, independently review cited passages and surrounding PDF context, resolve uncertain "
        "records, and verify source dates and reuse rights. Independent annotation and a representative test "
        "with positive and negative examples are needed for defensible performance estimates.",
        "",
    ])
    (output / "findings_and_limitations.md").write_text("\n".join(findings), encoding="utf-8")
    write_json(output / "results_metadata.json", {
        "basis": "User-reviewed annotations and non-independent AI assistant review of model flags.",
        "corpus_records": len(predictions), "model_flags": int(predictions["is_climate_related"].eq("True").sum()),
        "records_with_review_decisions": len(decisions),
        "esg_comparison": "Omitted; suitable dated scores were unavailable.",
        "limitations": [
            f"Convenience sample: {software_count} of {len(companies)} industry labels contain 'software'; "
            "industry composition and report availability limit generalization.",
            "Disclosure signals do not prove actual financial loss or a risk hidden from investors or ESG ratings.",
            "AI assistant review is not independent human annotation and may contain errors.",
            "Model-negative records were not comprehensively reviewed; no absence-of-risk conclusions.",
            "Model-positive selection and company counts do not estimate prevalence or financial impact.",
            "AI candidates do not have reviewed specificity or financial relevance labels.",
            "Random held-out sample has no positives; challenge is small, selected and previously inspected.",
            "Check PDF context and resolve uncertain fragments before using specific evidence in a paper.",
        ],
        "input_sha256": {
            "companies": _hash(companies_path), "predictions": _hash(predictions_path),
            "review_decisions": _hash(decisions_path), "source_checks": _hash(source_checks_path),
            "comparison": _hash(comparison_path), "manifest": _hash(manifest_path),
            "extraction_report": _hash(extraction_report_path),
            **{name: _hash(annotation_dir / name) for name in (
                "train_annotations.csv", "test_annotations.csv", "suggested_test_challenge_candidates.csv",
            )},
        },
    })
