import hashlib
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from pe_esg_analysis.annotation import LABEL_COLUMNS, RECORD_COLUMNS, prepare_annotation_set
from pe_esg_analysis.paper_results import generate_paper_results


@pytest.fixture
def inputs(tmp_path):
    companies = tmp_path / "companies.csv"
    corpus = tmp_path / "corpus.csv"
    annotations = tmp_path / "annotations"
    pd.DataFrame([
        {"company_id": name, "company_name": name, "acquisition_announcement_date": "2024-01-01"}
        for name in ("a", "b")
    ]).to_csv(companies, index=False)
    rows = pd.DataFrame([
        {"record_id": f"{company}{i}", "company_id": company, "document_id": company,
         "document_date": "2023-01-01", "source": "example", "page": str(i + 1),
         "text": f"{company} unique disclosure {i}"}
        for company in ("a", "b") for i in range(5)
    ])
    rows.to_csv(corpus, index=False)
    split = prepare_annotation_set(corpus, companies, annotations, per_company=2, test_companies=1)
    train = pd.read_csv(annotations / "train_annotations.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(annotations / "test_annotations.csv", dtype=str, keep_default_na=False)
    for frame in (train, test):
        frame["annotation_status"] = "complete"
        frame["is_climate_related"] = "no"
    train.loc[0, list(LABEL_COLUMNS)] = ["yes", "yes", "physical", "medium"]
    train.loc[1, ["annotation_status", "is_climate_related", "notes"]] = ["unusable", "", "fragment"]
    train_pool = rows[(rows.company_id == split["train_companies"][0]) & ~rows.record_id.isin(train.record_id)]
    candidates = train_pool.iloc[:2].copy()
    candidates["is_climate_related"] = "yes"
    candidates["is_specific"] = "yes"
    candidates["risk_category"] = ["transition", "regulatory"]
    candidates["financial_relevance"] = "low"
    candidates["annotation_status"] = "complete"
    candidates["notes"] = ""
    candidates.iloc[:1].to_csv(annotations / "suggested_training_candidates.csv", index=False)
    candidates.iloc[1:].to_csv(annotations / "suggested_training_candidates_round2.csv", index=False)
    train = pd.concat([train, candidates], ignore_index=True)
    train.to_csv(annotations / "train_annotations.csv", index=False)
    train[list(RECORD_COLUMNS)].to_csv(annotations / "train_disclosures.csv", index=False)
    test.to_csv(annotations / "test_annotations.csv", index=False)
    challenge = rows[(rows.company_id == split["test_companies"][0]) & ~rows.record_id.isin(test.record_id)].iloc[:1].copy()
    challenge["is_climate_related"] = "yes"
    challenge["is_specific"] = "yes"
    challenge["risk_category"] = "physical"
    challenge["financial_relevance"] = "medium"
    challenge["annotation_status"] = "complete"
    challenge["notes"] = ""
    challenge.to_csv(annotations / "suggested_test_challenge_candidates.csv", index=False)
    paths = {
        "companies": companies,
        "training_disclosures": annotations / "train_disclosures.csv",
        "training_annotations": annotations / "train_annotations.csv",
        "random_annotations": annotations / "test_annotations.csv",
        "challenge_annotations": annotations / "suggested_test_challenge_candidates.csv",
    }
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "metrics": [
            {"subset": subset, "model": model, "tn": 2 if subset == "random" else 0,
             "tp": 1 if subset == "challenge" else 0,
             "negative_support": 2 if subset == "random" else 0,
             "positive_support": 1 if subset == "challenge" else 0}
            for subset in ("random", "challenge") for model in ("ClimateBERT", "TF-IDF")
        ],
    }), encoding="utf-8")
    return {
        "companies_path": companies, "disclosures_path": corpus, "annotation_dir": annotations,
        "comparison_path": comparison, "output_dir": tmp_path / "results",
    }


def test_reviewed_results_keep_cohorts_and_provenance_separate(inputs):
    generate_paper_results(**inputs)
    output = inputs["output_dir"]
    summary = pd.read_csv(output / "annotation_cohort_summary.csv").set_index("cohort")
    assert summary.loc["train_random", ["reviewed", "unusable", "positive"]].tolist() == [1, 1, 1]
    assert summary.loc["train_candidates", ["reviewed", "positive"]].tolist() == [2, 2]
    assert summary.loc["train_candidates", ["positive_specific", "positive_relevance_low"]].tolist() == [2, 2]
    assert summary.loc["test_random", ["reviewed", "positive"]].tolist() == [2, 0]
    assert summary.loc["test_challenge", ["reviewed", "positive"]].tolist() == [1, 1]
    evidence = pd.read_csv(output / "reviewed_positive_evidence.csv")
    assert len(evidence) == 4 and set(evidence.cohort) == {"train_random", "train_candidates", "test_challenge"}
    assert set(RECORD_COLUMNS).issubset(evidence.columns)
    profiles = pd.read_csv(output / "reviewed_company_profiles.csv")
    assert len(profiles) == 2 and profiles["train_candidates_positive"].sum() == 2
    for name in ("reviewed_positive_categories.svg", "climate_detection_evaluation.svg"):
        ET.parse(output / name)
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_paper_results(**inputs)


def test_rejects_stale_comparison_or_unmerged_candidate(inputs):
    annotation_dir = inputs["annotation_dir"]
    comparison = json.loads(inputs["comparison_path"].read_text())
    comparison["input_sha256"]["random_annotations"] = "stale"
    inputs["comparison_path"].write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="stale"):
        generate_paper_results(**inputs)
    assert not inputs["output_dir"].exists()

    candidate_path = annotation_dir / "suggested_training_candidates.csv"
    candidate = pd.read_csv(candidate_path, keep_default_na=False)
    candidate.loc[0, "risk_category"] = "physical"
    candidate.to_csv(candidate_path, index=False)
    with pytest.raises(ValueError, match="differs"):
        generate_paper_results(**inputs)


def test_rejects_relabelled_random_sample(inputs):
    path = inputs["annotation_dir"] / "sampling_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["seed"] += 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="split|random cohort"):
        generate_paper_results(**inputs)
