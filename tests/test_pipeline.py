from datetime import date

import pytest

from pe_esg_analysis.classifiers import DisclosureClassifiers
from pe_esg_analysis.pipeline import ResearchPipeline, company_profiles, train_from_annotations
from pe_esg_analysis.preprocessing import select_evidence_sentence
from pe_esg_analysis.schemas import Company, Disclosure
from pe_esg_analysis.scoring import score_financial_relevance


def test_pipeline_respects_pre_acquisition_cutoff() -> None:
    companies = [Company("c1", "Example", date(2024, 1, 1))]
    disclosures = [
        Disclosure("r1", "c1", "d1", date(2023, 1, 1), "annual report", "4", "The facility faces flood risk and remediation costs of $4 million."),
        Disclosure("r2", "c1", "d2", date(2024, 2, 1), "annual report", "5", "The facility faces flood risk and remediation costs of $4 million."),
    ]
    predictions = ResearchPipeline().analyze(companies, disclosures)
    assert predictions[0].is_climate_related is True
    assert predictions[0].risk_category in {"physical", "environmental_liability"}
    assert predictions[0].financial_relevance == "medium"
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


def test_heuristics_recognize_weather_and_energy_disclosures():
    classifiers = DisclosureClassifiers()
    weather = "Hurricanes may interrupt our data centers."
    assert classifiers.detect_climate(weather)[0]
    assert classifiers.detect_specificity(weather)[0]
    assert classifiers.classify_risk(weather)[0] == "physical"
    assert classifiers.classify_risk("Our offices will operate on 100% renewable energy.")[0] == "transition"
    assert classifiers.classify_risk("Floods may cause litigation liability.")[0] == "physical"


def test_relevance_requires_financial_impact_not_specificity_or_percentages():
    def level(text: str) -> str:
        return score_financial_relevance(text, is_specific=True, risk_category="transition")[0]

    assert level("Our offices will operate on 100% renewable energy.") == "low"
    assert level("Flooding could interrupt services and increase costs.") == "medium"
    assert level("Flooding could have a material adverse effect on financial condition.") == "high"
    assert level("We have significant offices. Flooding could interrupt services.") == "medium"


def test_evidence_prefers_climate_risk_over_longer_unrelated_sentence():
    text = (
        "Our company has a long history of developing many services and products "
        "for customers across the world. Flooding threatens our facilities."
    )
    assert select_evidence_sentence(text) == "Flooding threatens our facilities."
    assert select_evidence_sentence("Flood risk. Business performance improved substantially.") == "Flood risk."
    assert select_evidence_sentence("No relevant sentence. Another unrelated sentence.") == "No relevant sentence."
