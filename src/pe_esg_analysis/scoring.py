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
    impact = re.compile(
        r"\b(?:costs?|expenses?|loss(?:es)?|lost|revenue|sales|profit|"
        r"financial condition|financial impact|operating results|cash flows?|lower demand|"
        r"litigation|penalt(?:y|ies)|fines?|impairment|liabilit(?:y|ies)|"
        r"disrupt(?:ion|ions|ed)?|interrupt(?:ion|ions|ed)?)\b",
        re.IGNORECASE,
    )
    material = re.compile(
        r"\b(?:material(?:ly)?\s+adverse|significant(?:ly)?\s+"
        r"(?:financial|monetary)\s+(?:impact|costs?|loss(?:es)?))\b",
        re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    has_impact = any(impact.search(sentence) for sentence in sentences)
    has_material_impact = any(
        impact.search(sentence) and material.search(sentence) for sentence in sentences
    )
    score = 5.0 if has_material_impact else 2.0 if has_impact else 0.0
    low, high = config.relevance_thresholds
    level = "high" if score >= high else "medium" if score >= low else "low"
    return level, score
