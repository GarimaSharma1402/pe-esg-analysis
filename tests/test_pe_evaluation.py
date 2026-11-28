import json

import pandas as pd
import pytest

from pe_esg_analysis.pe_evaluation import _allocate, _hash, evaluate_pe_review


def test_allocation_covers_both_strata():
    assert _allocate({"model_flagged": 2, "neither_flagged": 8}, 5) == {
        "model_flagged": 2, "neither_flagged": 3,
    }
    with pytest.raises(ValueError, match="Both"):
        _allocate({"model_flagged": 0, "neither_flagged": 8}, 5)


@pytest.fixture
def review_dir(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    reference = pd.DataFrame([
        {
            "record_id": str(i), "company_id": "c", "document_id": "doc",
            "document_date": "2021-01-01", "source": "source", "page": str(i), "text": f"text {i}",
        }
        for i in range(4)
    ])
    reference.to_csv(review / "sample_reference.csv", index=False)
    frozen = pd.DataFrame([
        {"record_id": "0", "stratum": "model_flagged", "climatebert": True, "tfidf": True, "population_size": 2, "sample_size": 2},
        {"record_id": "1", "stratum": "model_flagged", "climatebert": False, "tfidf": True, "population_size": 2, "sample_size": 2},
        {"record_id": "2", "stratum": "neither_flagged", "climatebert": False, "tfidf": False, "population_size": 8, "sample_size": 2},
        {"record_id": "3", "stratum": "neither_flagged", "climatebert": False, "tfidf": False, "population_size": 8, "sample_size": 2},
    ])
    frozen.to_csv(review / "frozen_predictions_and_design.csv", index=False)
    labels = reference.copy()
    labels["annotation_status"] = "complete"
    labels["is_climate_related"] = ["yes", "no", "yes", "no"]
    labels["notes"] = ""
    labels.to_csv(review / "review_sheet.csv", index=False)
    (review / "sampling_metadata.json").write_text(json.dumps({
        "reference_sha256": _hash(review / "sample_reference.csv"),
        "frozen_design_sha256": _hash(review / "frozen_predictions_and_design.csv"),
        "stratum_populations": {"model_flagged": 2, "neither_flagged": 8},
        "stratum_sample_sizes": {"model_flagged": 2, "neither_flagged": 2},
        "limitations": [],
    }), encoding="utf-8")
    return review


def test_weighted_metrics_use_sampling_probabilities(review_dir, tmp_path):
    output = tmp_path / "results"
    evaluate_pe_review(review_dir, output)
    result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    climatebert = result["metrics"][0]
    assert (climatebert["weighted_tp"], climatebert["weighted_fn"], climatebert["weighted_fp"], climatebert["weighted_tn"]) == (1, 4, 0, 5)
    assert climatebert["accuracy"] == pytest.approx(0.6)
    assert climatebert["positive_recall"] == pytest.approx(0.2)
    assert climatebert["positive_f1"] == pytest.approx(1 / 3)
    assert result["metrics"][1]["positive_f1"] == pytest.approx(2 / 7)


def test_incomplete_or_modified_review_is_rejected(review_dir, tmp_path):
    path = review_dir / "review_sheet.csv"
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    labels.loc[0, "annotation_status"] = ""
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Every record"):
        evaluate_pe_review(review_dir, tmp_path / "incomplete")
    labels.loc[0, "annotation_status"] = "complete"
    labels.loc[0, "text"] = "changed"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source text"):
        evaluate_pe_review(review_dir, tmp_path / "modified")


def test_unusable_records_are_not_counted_as_negative(review_dir, tmp_path):
    path = review_dir / "review_sheet.csv"
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    labels.loc[0, ["annotation_status", "is_climate_related", "notes"]] = ["unusable", "", "Fragment"]
    labels.to_csv(path, index=False)
    output = tmp_path / "results"
    evaluate_pe_review(review_dir, output)
    result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert result["unusable_records"] == 1
    assert result["metrics"][0]["reviewed_positives"] == 1
    assert result["metrics"][0]["weighted_tp"] == 0
