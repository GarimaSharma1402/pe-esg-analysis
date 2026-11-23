from __future__ import annotations

import re

from .config import PipelineConfig


def score_financial_relevance(
    text: str,
    *,
    is_specific: bool,
    risk_category: str,
    config: PipelineConfig | None = None,
) -> tuple[str, float]:
    """Score only observable disclosure signals; this is not a financial forecast."""
    config = config or PipelineConfig()
    lowered = text.casefold()
    score = 0.0
    if is_specific:
        score += 2.0
    if re.search(r"\d+(?:\.\d+)?\s*%|\$[\d,.]+|\b(?:million|billion|tonnes?)\b", lowered):
        score += 2.0
    if any(term in lowered for term in ("cost", "loss", "impairment", "capital expenditure", "revenue", "insurance")):
        score += 2.0
    if any(term in lowered for term in ("litigation", "penalty", "fine", "non-compliance", "stranded asset")):
        score += 2.0
    if risk_category in {"regulatory", "environmental_liability"}:
        score += 1.0
    low, high = config.relevance_thresholds
    level = "high" if score >= high else "medium" if score >= low else "low"
    return level, score
