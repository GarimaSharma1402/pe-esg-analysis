from __future__ import annotations

import re
from collections.abc import Iterable


_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_text(text: str) -> str:
    """Normalize extracted text while retaining readable punctuation."""
    text = text.replace("\u00ad", "").replace("-\n", "")
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str, min_length: int = 20) -> list[str]:
    cleaned = clean_text(text)
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(cleaned)
        if len(sentence.strip()) >= min_length
    ]


def climate_terms(text: str, keywords: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    return [keyword for keyword in keywords if keyword.casefold() in lowered]


_EVIDENCE_CLIMATE = re.compile(
    r"\b(?:(?<!business )climate|greenhouse|ghg|carbon|emissions?|"
    r"renewable energy|decarbon\w*|electrification|flood\w*|drought\w*|"
    r"wildfires?|hurricanes?|storms?|tornado(?:es|s)?|blizzards?|heatwaves?|"
    r"extreme weather|sea level)\b",
    re.IGNORECASE,
)
_EVIDENCE_IMPACT = re.compile(
    r"\b(?:risks?|threats?|exposures?|costs?|expenses?|loss(?:es)?|revenue|"
    r"financial|material(?:ly)?|disrupt\w*|interrupt\w*|regulat\w*|compliance)\b",
    re.IGNORECASE,
)


def select_evidence_sentence(text: str, min_length: int = 20) -> str:
    """Prefer a climate-related sentence with a stated risk or impact."""
    sentences = split_sentences(text, min_length=1)
    if not sentences:
        raise ValueError("Cannot select evidence from empty text")
    eligible = [
        sentence for sentence in sentences
        if len(sentence) >= min_length or _EVIDENCE_CLIMATE.search(sentence)
    ] or sentences
    return max(
        eligible,
        key=lambda sentence: (
            bool(_EVIDENCE_CLIMATE.search(sentence)),
            bool(_EVIDENCE_IMPACT.search(sentence)),
        ),
    )
