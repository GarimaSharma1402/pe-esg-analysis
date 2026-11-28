import pandas as pd
import pytest

from pe_esg_analysis.evaluation import evaluate_predictions
from pe_esg_analysis.comparison import climate_metrics, compare_climate_detectors


def frames():
    actual = pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "is_climate_related": ["yes", "no", "1"],
        "is_specific": ["", "", ""],
        "risk_category": ["", "", ""],
        "financial_relevance": ["", "", ""],
    })
    predictions = pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "is_before_cutoff": [True, True, True],
        "is_climate_related": [True, True, False],
        "is_specific": [False, False, False],
        "risk_category": ["other", "other", "other"],
        "financial_relevance": ["low", "low", "low"],
    })
    return actual, predictions


def test_metrics_normalize_binary_and_exclude_missing_annotations():
    actual, predictions = frames()
    result = evaluate_predictions(actual, predictions)
    metrics = result["climate_detection"]
    assert list(result) == ["climate_detection"]
    assert metrics["labels"] == ["no", "yes"]
    assert metrics["confusion_matrix"] == [[0, 1], [1, 1]]
    assert metrics["report"]["yes"]["f1-score"] == pytest.approx(0.5)
    assert metrics["f1_macro"] == pytest.approx(0.25)


def test_evaluation_rejects_missing_duplicate_and_post_cutoff_records():
    actual, predictions = frames()
    with pytest.raises(ValueError, match="without predictions"):
        evaluate_predictions(actual, predictions.iloc[:2])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_predictions(pd.concat([actual, actual.iloc[:1]]), predictions)
    predictions.loc[0, "is_before_cutoff"] = False
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_predictions(actual, predictions)


def test_comparison_metrics_do_not_report_single_class_precision_or_f1():
    negative = climate_metrics([False, False], [False, True])
    assert negative["positive_recall"] is None
    assert negative["positive_precision"] is None
    assert negative["positive_f1"] is None
    assert negative["false_positive_rate"] == 0.5
    positive = climate_metrics([True, True], [True, False])
    assert positive["positive_recall"] == 0.5
    assert positive["positive_precision"] is None
    assert positive["positive_f1"] is None
    mixed = climate_metrics([False, True, True], [True, True, False])
    assert mixed["positive_precision"] == 0.5
    assert mixed["positive_recall"] == 0.5
    assert mixed["positive_f1"] == 0.5


def test_comparison_uses_reviewed_heldout_companies_without_overwriting(tmp_path, monkeypatch):
    class FakeDetector:
        def predict_one(self, text):
            return ("flood" in text.lower(), 0.9)

    monkeypatch.setattr("pe_esg_analysis.comparison.ClimateBertDetector", FakeDetector)
    companies = pd.DataFrame([
        {"company_id": name, "company_name": name, "acquisition_announcement_date": "2024-01-01"}
        for name in ("train", "random", "challenge")
    ])

    def row(record_id, company, text, climate, specific="", risk="", relevance=""):
        return {
            "record_id": record_id, "company_id": company, "document_id": company,
            "document_date": "2023-01-01", "source": "source", "page": "1", "text": text,
            "is_climate_related": climate, "is_specific": specific, "risk_category": risk,
            "financial_relevance": relevance, "annotation_status": "complete", "notes": "",
        }

    train = pd.DataFrame([
        row("t1", "train", "Annual revenue increased.", "no"),
        row("t2", "train", "Flooding threatens our facility.", "yes", "yes", "physical", "medium"),
        row("t3", "train", "We support climate action.", "yes", "no", "transition", "low"),
    ])
    random = pd.DataFrame([row("r1", "random", "This company sells software.", "no")])
    challenge = pd.DataFrame([
        row("c1", "challenge", "Flood risk threatens our offices.", "yes", "yes", "physical", "medium")
    ])
    paths = {name: tmp_path / f"{name}.csv" for name in ("companies", "train", "random", "challenge")}
    for name, frame in (("companies", companies), ("train", train), ("random", random), ("challenge", challenge)):
        frame.to_csv(paths[name], index=False)
    output = tmp_path / "result"
    kwargs = {
        "companies_path": paths["companies"], "training_disclosures_path": paths["train"],
        "training_annotations_path": paths["train"], "random_annotations_path": paths["random"],
        "challenge_annotations_path": paths["challenge"], "output_dir": output,
    }
    compare_climate_detectors(**kwargs)
    results = pd.read_csv(output / "comparison.csv")
    assert len(results) == 4
    assert results["positive_precision"].isna().all()
    assert pd.read_csv(output / "paired_predictions.csv")["record_id"].tolist() == ["r1", "c1"]
    with pytest.raises(FileExistsError, match="overwrite"):
        compare_climate_detectors(**kwargs)
    random.loc[0, "company_id"] = "train"
    random.to_csv(paths["random"], index=False)
    kwargs["output_dir"] = tmp_path / "other"
    with pytest.raises(ValueError, match="Held-out company"):
        compare_climate_detectors(**kwargs)
    random.loc[0, "company_id"] = "random"
    random.loc[0, "text"] = "  annual   REVENUE increased. "
    random.to_csv(paths["random"], index=False)
    with pytest.raises(ValueError, match="text overlap"):
        compare_climate_detectors(**kwargs)
