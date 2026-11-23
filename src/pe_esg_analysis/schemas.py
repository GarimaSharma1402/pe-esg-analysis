from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


RISK_CATEGORIES = (
    "physical",
    "transition",
    "regulatory",
    "environmental_liability",
    "other",
)
RELEVANCE_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Company:
    company_id: str
    company_name: str
    acquisition_announcement_date: date
    industry: str = ""
    esg_score: float | None = None


@dataclass(frozen=True)
class Disclosure:
    record_id: str
    company_id: str
    document_id: str
    document_date: date
    source: str
    page: str
    text: str
    section: str = ""


@dataclass
class Prediction:
    record_id: str
    company_id: str
    document_id: str
    document_date: date
    source: str
    page: str
    section: str
    text: str
    is_before_cutoff: bool
    is_climate_related: bool
    climate_confidence: float
    is_specific: bool
    specificity_confidence: float
    risk_category: str
    risk_confidence: float
    financial_relevance: str
    relevance_score: float
    evidence_sentence: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
