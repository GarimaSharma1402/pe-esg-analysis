import pandas as pd
import pytest

from pe_esg_analysis.evaluation import evaluate_predictions


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
