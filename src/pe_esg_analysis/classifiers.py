from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import PipelineConfig
from .preprocessing import climate_terms


class ClimateDetector(Protocol):
    def predict_one(self, text: str) -> tuple[bool, float]: ...


@dataclass
class TextModel:
    """Small, reproducible supervised baseline using TF-IDF and logistic regression."""

    model: Pipeline | None = None
    classes: tuple[str, ...] = ()

    def fit(self, texts: list[str], labels: list[str]) -> "TextModel":
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and have equal length")
        if len(set(labels)) < 2:
            raise ValueError("supervised training requires at least two label classes")
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.model.fit(texts, labels)
        self.classes = tuple(str(value) for value in self.model.classes_)
        return self

    def predict_one(self, text: str) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        probabilities = self.model.predict_proba([text])[0]
        index = int(probabilities.argmax())
        return str(self.classes[index]), float(probabilities[index])


class DisclosureClassifiers:
    """Detectors with heuristic fallback and optional supervised models."""

    def __init__(
        self, config: PipelineConfig | None = None, climate_detector: ClimateDetector | None = None
    ) -> None:
        self.config = config or PipelineConfig()
        self.climate_detector = climate_detector
        self.climate_model = TextModel()
        self.specificity_model = TextModel()
        self.risk_model = TextModel()

    def detect_climate(self, text: str) -> tuple[bool, float]:
        if self.climate_detector is not None:
            return self.climate_detector.predict_one(text)
        if self.climate_model.model is not None:
            label, confidence = self.climate_model.predict_one(text)
            return label.casefold() in {"yes", "true", "1", "climate"}, confidence
        terms = climate_terms(text, self.config.climate_keywords)
        confidence = min(1.0, len(terms) / 3)
        return bool(terms), confidence

    def detect_specificity(self, text: str) -> tuple[bool, float]:
        if self.specificity_model.model is not None:
            label, confidence = self.specificity_model.predict_one(text)
            return label.casefold() in {"yes", "true", "1", "specific"}, confidence
        specific_markers = (
            r"\d+(?:\.\d+)?\s*%",
            r"\$[\d,.]+",
            r"\b(?:tonnes?|metric tons?|mw|mwh|tco2e)\b",
            r"\b(?:facility|plant|asset|supplier|site|region)\b",
        )
        matches = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in specific_markers)
        return matches > 0, min(1.0, matches / 2)

    def classify_risk(self, text: str) -> tuple[str, float]:
        if self.risk_model.model is not None:
            return self.risk_model.predict_one(text)
        lowered = text.casefold()
        keyword_groups = {
            "physical": ("flood", "drought", "wildfire", "storm", "heat", "sea level"),
            "transition": ("decarbon", "net zero", "stranded", "energy transition", "carbon price"),
            "regulatory": ("regulation", "regulatory", "compliance", "law", "disclosure", "permit"),
            "environmental_liability": ("remediation", "contamination", "spill", "cleanup", "liability"),
        }
        matches = {
            category: sum(term in lowered for term in terms)
            for category, terms in keyword_groups.items()
        }
        category, count = max(matches.items(), key=lambda item: item[1])
        if count == 0:
            return "other", 0.25
        return category, min(1.0, 0.5 + count * 0.15)
