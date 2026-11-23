from __future__ import annotations

from .classifiers import DisclosureClassifiers
from .config import PipelineConfig
from .preprocessing import clean_text, split_sentences
from .schemas import Company, Disclosure, Prediction
from .scoring import score_financial_relevance


class ResearchPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        classifiers: DisclosureClassifiers | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.classifiers = classifiers or DisclosureClassifiers(self.config)

    def analyze(self, companies: list[Company], disclosures: list[Disclosure]) -> list[Prediction]:
        companies_by_id = {company.company_id: company for company in companies}
        predictions: list[Prediction] = []
        for disclosure in disclosures:
            company = companies_by_id.get(disclosure.company_id)
            if company is None:
                raise ValueError(f"Unknown company_id: {disclosure.company_id}")
            before_cutoff = disclosure.document_date < company.acquisition_announcement_date
            text = clean_text(disclosure.text)
            if not text:
                raise ValueError(f"Empty disclosure text for record_id: {disclosure.record_id}")
            try:
                climate, climate_confidence = self.classifiers.detect_climate(text)
            except ValueError as exc:
                raise ValueError(f"Climate detection failed for record_id {disclosure.record_id}: {exc}") from exc
            specific, specificity_confidence = self.classifiers.detect_specificity(text)
            category, risk_confidence = self.classifiers.classify_risk(text)
            relevance, relevance_score = score_financial_relevance(
                text, is_specific=specific, risk_category=category, config=self.config
            )
            evidence = split_sentences(text, self.config.min_sentence_length)
            evidence_sentence = max(evidence, key=len, default=text)
            predictions.append(Prediction(
                record_id=disclosure.record_id,
                company_id=disclosure.company_id,
                document_id=disclosure.document_id,
                document_date=disclosure.document_date,
                source=disclosure.source,
                page=disclosure.page,
                section=disclosure.section,
                text=text,
                is_before_cutoff=before_cutoff,
                is_climate_related=climate and before_cutoff,
                climate_confidence=climate_confidence,
                is_specific=specific and climate and before_cutoff,
                specificity_confidence=specificity_confidence,
                risk_category=category if climate and before_cutoff else "other",
                risk_confidence=risk_confidence,
                financial_relevance=relevance if climate and before_cutoff else "low",
                relevance_score=relevance_score if climate and before_cutoff else 0.0,
                evidence_sentence=evidence_sentence,
            ))
        return predictions


def train_from_annotations(
    pipeline: ResearchPipeline,
    disclosures: list[Disclosure],
    annotations: list[dict[str, str]],
) -> ResearchPipeline:
    """Fit supervised models from manually reviewed disclosure records."""
    by_id = {disclosure.record_id: disclosure for disclosure in disclosures}
    rows = [row for row in annotations if row.get("record_id") in by_id]
    if not rows:
        raise ValueError("No annotations match the supplied disclosures")
    texts = [by_id[row["record_id"]].text for row in rows]
    pipeline.classifiers.climate_model.fit(
        texts, [str(row["is_climate_related"]) for row in rows]
    )
    pipeline.classifiers.specificity_model.fit(
        texts, [str(row["is_specific"]) for row in rows]
    )
    pipeline.classifiers.risk_model.fit(
        texts, [str(row["risk_category"]) for row in rows]
    )
    return pipeline


def company_profiles(predictions: list[Prediction]) -> list[dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for prediction in predictions:
        profile = profiles.setdefault(prediction.company_id, {
            "company_id": prediction.company_id,
            "eligible_disclosures": 0,
            "climate_evidence_count": 0,
            "specific_evidence_count": 0,
            "risk_category_counts": {},
            "relevance_counts": {"low": 0, "medium": 0, "high": 0},
            "total_relevance_score": 0.0,
        })
        if not prediction.is_before_cutoff:
            continue
        profile["eligible_disclosures"] = int(profile["eligible_disclosures"]) + 1
        if prediction.is_climate_related:
            profile["climate_evidence_count"] = int(profile["climate_evidence_count"]) + 1
            profile["risk_category_counts"][prediction.risk_category] = (
                profile["risk_category_counts"].get(prediction.risk_category, 0) + 1
            )
            profile["relevance_counts"][prediction.financial_relevance] += 1
            profile["total_relevance_score"] = (
                float(profile["total_relevance_score"]) + prediction.relevance_score
            )
        if prediction.is_specific:
            profile["specific_evidence_count"] = int(profile["specific_evidence_count"]) + 1
    return list(profiles.values())
