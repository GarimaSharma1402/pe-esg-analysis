"""Compare climate detectors on the same separately reported held-out cohorts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from .annotation import RECORD_COLUMNS, reviewed_annotations
from .classifiers import DisclosureClassifiers
from .climatebert import MODEL_ID, MODEL_REVISION, ClimateBertDetector
from .io import load_companies, load_disclosures, write_json
from .pipeline import ResearchPipeline, train_from_annotations


def climate_metrics(actual: list[bool], predicted: list[bool]) -> dict[str, int | float | None]:
    """Report binary counts; suppress precision/F1 for single-class cohorts."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Climate labels and predictions must be non-empty and equal length")
    tp = sum(a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    positive = tp + fn
    negative = tn + fp
    both_classes = positive > 0 and negative > 0
    return {
        "n": len(actual),
        "positive_support": positive,
        "negative_support": negative,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "positive_precision": tp / (tp + fp) if both_classes and tp + fp else None,
        "positive_recall": tp / positive if positive else None,
        "positive_f1": 2 * tp / (2 * tp + fp + fn) if both_classes else None,
        "false_positive_rate": fp / negative if negative else None,
        "negative_correct_rate": tn / negative if negative else None,
    }


def _read_reviewed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = (set(RECORD_COLUMNS) | {"annotation_status"}) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return reviewed_annotations(frame)


def compare_climate_detectors(
    *,
    companies_path: str | Path,
    training_disclosures_path: str | Path,
    training_annotations_path: str | Path,
    random_annotations_path: str | Path,
    challenge_annotations_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Run both detectors and keep random and selected challenge metrics separate."""
    paths = {
        "companies": Path(companies_path),
        "training_disclosures": Path(training_disclosures_path),
        "training_annotations": Path(training_annotations_path),
        "random_annotations": Path(random_annotations_path),
        "challenge_annotations": Path(challenge_annotations_path),
    }
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {output}")

    companies = load_companies(paths["companies"])
    company_by_id = {company.company_id: company for company in companies}
    if len(company_by_id) != len(companies):
        raise ValueError("Company IDs must be unique")
    training = _read_reviewed(paths["training_annotations"])
    train_disclosures = load_disclosures(paths["training_disclosures"])
    train_by_id = {record.record_id: record for record in train_disclosures}
    if len(train_by_id) != len(train_disclosures):
        raise ValueError("Training disclosure IDs must be unique")
    train_rows = training.to_dict("records")
    for row in train_rows:
        record = train_by_id.get(row["record_id"])
        if record is None:
            raise ValueError(f"Missing training disclosure: {row['record_id']}")
        for column in RECORD_COLUMNS[1:]:
            if str(getattr(record, column)) != row[column]:
                raise ValueError(f"Training annotation mismatch: {row['record_id']} {column}")

    cohorts = {
        "random": _read_reviewed(paths["random_annotations"]),
        "challenge": _read_reviewed(paths["challenge_annotations"]),
    }
    heldout_records = {
        name: {record.record_id: record for record in load_disclosures(paths[f"{name}_annotations"])}
        for name in cohorts
    }
    train_companies = {record.company_id for record in train_disclosures}
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    train_texts = {normalize_text(record.text) for record in train_disclosures}
    seen_ids = set(train_by_id)
    for name, frame in cohorts.items():
        records = heldout_records[name]
        if len(records) != len(pd.read_csv(paths[f"{name}_annotations"], dtype=str)):
            raise ValueError(f"Duplicate disclosure IDs in {name}")
        for _, row in frame.iterrows():
            record = records[row["record_id"]]
            if record.record_id in seen_ids:
                raise ValueError(f"Overlapping record ID: {record.record_id}")
            if record.company_id in train_companies:
                raise ValueError(f"Held-out company appears in training: {record.company_id}")
            if normalize_text(record.text) in train_texts:
                raise ValueError(f"Exact held-out/training text overlap: {record.record_id}")
            seen_ids.add(record.record_id)

    for record in train_disclosures + [
        heldout_records[name][record_id]
        for name, frame in cohorts.items() for record_id in frame["record_id"]
    ]:
        company = company_by_id.get(record.company_id)
        if company is None:
            raise ValueError(f"Unknown company ID: {record.company_id}")
        if record.document_date >= company.acquisition_announcement_date:
            raise ValueError(f"Disclosure is on or after acquisition: {record.record_id}")

    training_pipeline = train_from_annotations(ResearchPipeline(), train_disclosures, train_rows)
    climate_pipeline = ResearchPipeline(
        classifiers=DisclosureClassifiers(climate_detector=ClimateBertDetector())
    )
    pipelines = {"ClimateBERT": climate_pipeline, "TF-IDF": training_pipeline}
    results = []
    paired = []
    for name, frame in cohorts.items():
        records = [heldout_records[name][record_id] for record_id in frame["record_id"]]
        actual = frame["is_climate_related"].str.strip().str.lower().map(
            {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
        )
        if actual.isna().any():
            raise ValueError(f"Invalid climate labels in {name}")
        predictions = {
            model: pipeline.analyze(companies, records) for model, pipeline in pipelines.items()
        }
        for model, values in predictions.items():
            if [value.record_id for value in values] != frame["record_id"].tolist():
                raise ValueError(f"Prediction record IDs do not match {name}")
            results.append({
                "subset": name,
                "model": model,
                **climate_metrics(actual.tolist(), [value.is_climate_related for value in values]),
            })
        for index, (_, row) in enumerate(frame.iterrows()):
            paired.append({
                "subset": name,
                "record_id": row["record_id"],
                "company_id": row["company_id"],
                "actual_climate": "yes" if actual.iloc[index] else "no",
                **{model: "yes" if values[index].is_climate_related else "no"
                   for model, values in predictions.items()},
            })

    output.mkdir(parents=True)
    write_json(output / "comparison.json", {
        "task": "climate relevance detection",
        "climatebert_model": MODEL_ID,
        "climatebert_revision": MODEL_REVISION,
        "training_reviewed": len(training),
        "training_positive": int(
            training["is_climate_related"].str.strip().str.lower().isin(["yes", "true", "1"]).sum()
        ),
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "limitations": [
            "Report random and enriched challenge cohorts separately; challenge selection biases precision and F1.",
            "Single-class cohort precision and F1 are suppressed; recall requires positive support.",
            "Company and exact-text split checks do not exclude near-duplicates.",
            "Downstream classifiers differ between pipelines; only climate detection is compared.",
        ],
        "metrics": results,
    })
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_predictions.csv", index=False)
