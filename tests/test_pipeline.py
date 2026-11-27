from datetime import date

import pytest

from pe_esg_analysis.pipeline import ResearchPipeline, company_profiles, train_from_annotations
from pe_esg_analysis.schemas import Company, Disclosure


def test_pipeline_respects_pre_acquisition_cutoff() -> None:
    companies = [Company("c1", "Example", date(2024, 1, 1))]
    disclosures = [
        Disclosure("r1", "c1", "d1", date(2023, 1, 1), "annual report", "4", "The facility faces flood risk and remediation costs of $4 million."),
        Disclosure("r2", "c1", "d2", date(2024, 2, 1), "annual report", "5", "The facility faces flood risk and remediation costs of $4 million."),
    ]
    predictions = ResearchPipeline().analyze(companies, disclosures)
    assert predictions[0].is_climate_related is True
    assert predictions[0].risk_category in {"physical", "environmental_liability"}
    assert predictions[0].financial_relevance == "high"
    assert predictions[1].is_climate_related is False
    assert company_profiles(predictions)[0]["climate_evidence_count"] == 1


def test_generic_climate_statement_is_low_relevance() -> None:
    companies = [Company("c1", "Example", date(2024, 1, 1))]
    disclosure = Disclosure("r1", "c1", "d1", date(2023, 1, 1), "esg report", "2", "We support climate action and sustainability.")
    prediction = ResearchPipeline().analyze(companies, [disclosure])[0]
    assert prediction.is_climate_related is True
    assert prediction.is_specific is False
    assert prediction.financial_relevance == "low"


def test_training_ignores_blank_downstream_labels():
    disclosures = [
        Disclosure("a", "c", "d", date(2023, 1, 1), "source", "1", "Revenue increased."),
        Disclosure("b", "c", "d", date(2023, 1, 1), "source", "2", "Flooding damaged the facility."),
        Disclosure("c", "c", "d", date(2023, 1, 1), "source", "3", "We discuss climate policy."),
    ]
    rows = [
        {"record_id": "a", "is_climate_related": "no", "is_specific": "", "risk_category": ""},
        {"record_id": "b", "is_climate_related": "yes", "is_specific": "yes", "risk_category": "physical"},
        {"record_id": "c", "is_climate_related": "yes", "is_specific": "no", "risk_category": "regulatory"},
    ]
    pipeline = train_from_annotations(ResearchPipeline(), disclosures, rows)
    assert set(pipeline.classifiers.specificity_model.classes) == {"yes", "no"}
    assert set(pipeline.classifiers.risk_model.classes) == {"physical", "regulatory"}
    rows[2]["risk_category"] = ""
    with pytest.raises(ValueError, match="risk_category training requires"):
        train_from_annotations(ResearchPipeline(), disclosures, rows)
