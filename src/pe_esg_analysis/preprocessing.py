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
