"""Run offline, batched ClimateBERT screening with per-company checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .classifiers import DisclosureClassifiers
from .climatebert import MODEL_ID, MODEL_REVISION, ClimateBertDetector
from .io import load_companies, load_disclosures, write_json, write_predictions
from .pipeline import ResearchPipeline, company_profiles
from .preprocessing import clean_text


class _CachedDetector:
    def __init__(self, predictions: dict[str, tuple[bool, float]]) -> None:
        self.predictions = predictions

    def predict_one(self, text: str) -> tuple[bool, float]:
        return self.predictions[text]


def run_full_corpus(
    companies_path: str | Path, disclosures_path: str | Path,
    output_dir: str | Path, batch_size: int = 32,
) -> None:
    companies_path, disclosures_path, output = Path(companies_path), Path(disclosures_path), Path(output_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results directory: {output}")
    companies = load_companies(companies_path)
    disclosures = load_disclosures(disclosures_path)
    by_company = {company.company_id: company for company in companies}
    if len(by_company) != len(companies) or not disclosures:
        raise ValueError("Companies must have unique IDs and disclosures must be non-empty")
    if len({record.record_id for record in disclosures}) != len(disclosures):
        raise ValueError("Disclosure record IDs must be unique")
    if {record.company_id for record in disclosures} != set(by_company):
        raise ValueError("Company metadata and disclosure company IDs do not match")
    for record in disclosures:
        if record.document_date >= by_company[record.company_id].acquisition_announcement_date:
            raise ValueError(f"Disclosure is on or after acquisition announcement: {record.record_id}")
        if not clean_text(record.text):
            raise ValueError(f"Empty disclosure: {record.record_id}")
    detector = ClimateBertDetector()
    metadata = {
        "status": "in_progress", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "climate_detector": "ClimateBERT", "other_classifiers": "exploratory rules",
        "batch_size": batch_size, "disclosures": len(disclosures), "companies": len(companies),
        "input_sha256": {
            "companies": hashlib.sha256(companies_path.read_bytes()).hexdigest(),
            "disclosures": hashlib.sha256(disclosures_path.read_bytes()).hexdigest(),
        },
        "limitations": [
            "Predictions are unreviewed and may miss or misclassify relevant disclosures.",
            "Counts are extracted text blocks, not independent risks or financial impacts.",
            "Specificity, risk categories, relevance and sentence selection use exploratory rules.",
            "Batched inference can differ slightly from individual inference in floating-point confidence.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "run_metadata.json", metadata)
    cache: dict[str, tuple[bool, float]] = {}
    predictions = []
    for company in companies:
        records = [record for record in disclosures if record.company_id == company.company_id]
        texts = list(dict.fromkeys(clean_text(record.text) for record in records))
        missing = [text for text in texts if text not in cache]
        cache.update(zip(missing, detector.predict_many(missing, batch_size), strict=True))
        pipeline = ResearchPipeline(classifiers=DisclosureClassifiers(climate_detector=_CachedDetector(cache)))
        company_predictions = pipeline.analyze(companies, records)
        write_predictions(output / "by_company" / f"{company.company_id}.csv", company_predictions)
        predictions.extend(company_predictions)
        print(f"{company.company_id}: analyzed {len(records)} records", flush=True)
    write_predictions(output / "predictions.csv", predictions)
    profiles = company_profiles(predictions)
    for profile in profiles:
        profile.pop("total_relevance_score")
        profile["model_climate_positive_records"] = profile.pop("climate_evidence_count")
        profile["rule_specific_positive_records"] = profile.pop("specific_evidence_count")
        profile["rule_category_counts"] = json.dumps(profile.pop("risk_category_counts"), sort_keys=True)
        profile["rule_relevance_counts"] = json.dumps(profile.pop("relevance_counts"), sort_keys=True)
    pd.DataFrame(profiles).to_csv(output / "company_profiles.csv", index=False)
    metadata["status"] = "complete"
    metadata["unique_cleaned_texts"] = len(cache)
    write_json(output / "run_metadata.json", metadata)
