from datetime import date
from types import SimpleNamespace
from xml.etree import ElementTree

import pandas as pd
import pytest

from pe_esg_analysis.reviewed_results import _profiles, _review_figure, validate_reviews


@pytest.fixture
def reviewed_inputs():
    predictions = pd.DataFrame([
        {"record_id": record_id, "company_id": "c", "is_climate_related": flag}
        for record_id, flag in (("positive", "True"), ("negative", "True"), ("candidate", "True"), ("missed", "False"))
    ])
    annotations = pd.DataFrame([
        {"record_id": record_id, "annotation_status": "complete", "is_climate_related": label,
         "is_specific": "yes" if label == "yes" else "", "risk_category": "physical" if label == "yes" else "",
         "financial_relevance": "medium" if label == "yes" else "", "notes": ""}
        for record_id, label in (("positive", "yes"), ("negative", "no"), ("missed", "yes"))
    ])
    decisions = pd.DataFrame([
        {"record_id": record_id, "company_id": "c", "pdf": "c.pdf", "model_climate_positive": flag,
         "review_outcome": outcome, "review_category": category, "user_specificity": specific,
         "user_financial_relevance": relevance}
        for record_id, flag, outcome, category, specific, relevance in (
            ("positive", "True", "user_climate_positive", "physical", "yes", "medium"),
            ("negative", "True", "user_not_climate", "", "", ""),
            ("candidate", "True", "ai_climate_candidate", "transition", "", ""),
            ("missed", "False", "user_climate_positive", "physical", "yes", "medium"),
        )
    ])
    checks = pd.DataFrame([
        {"record_id": record_id, "company_id": "c", "pdf": "c.pdf",
         "pdf_block_match": "True", "manifest_match": "True"}
        for record_id in predictions["record_id"]
    ])
    return predictions, decisions, checks, annotations


def test_review_coverage_includes_user_positive_missed_by_model(reviewed_inputs):
    predictions, decisions, checks, annotations = reviewed_inputs
    result = validate_reviews(predictions, decisions, checks, annotations)
    assert len(result) == 4
    with pytest.raises(ValueError, match="cover"):
        validate_reviews(predictions, decisions[decisions.record_id != "missed"], checks, annotations)


def test_user_labels_and_source_checks_cannot_be_replaced(reviewed_inputs):
    predictions, decisions, checks, annotations = reviewed_inputs
    decisions.loc[decisions.record_id == "negative", "review_outcome"] = "ai_not_climate_in_record"
    with pytest.raises(ValueError, match="user-negative"):
        validate_reviews(predictions, decisions, checks, annotations)
    decisions.loc[decisions.record_id == "negative", "review_outcome"] = "user_not_climate"
    checks.loc[checks.record_id == "candidate", "pdf"] = "other.pdf"
    with pytest.raises(ValueError, match="PDF"):
        validate_reviews(predictions, decisions, checks, annotations)


def test_profiles_separate_review_bases_and_exclude_misses_from_figure(reviewed_inputs):
    predictions, decisions, checks, annotations = reviewed_inputs
    reviewed = validate_reviews(predictions, decisions, checks, annotations)
    companies = [SimpleNamespace(company_id="c", company_name="C & Co", acquisition_announcement_date=date(2023, 1, 1))]
    profiles = _profiles(companies, predictions, reviewed)
    row = profiles.iloc[0]
    assert (row["model_flagged_records"], row["user_climate_positive"], row["ai_climate_candidate"]) == (3, 1, 1)
    assert (row["user_positive_model_missed"], row["user_climate_positive_physical"]) == (1, 2)
    figure = _review_figure(profiles)
    ElementTree.fromstring(figure)
    assert "1 user-reviewed positives missed" in figure
