from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score,
)

from .schemas import RELEVANCE_LEVELS, RISK_CATEGORIES


def evaluate_labels(actual: Iterable[str], predicted: Iterable[str]) -> dict[str, Any]:
    actual_list, predicted_list = list(actual), list(predicted)
    if len(actual_list) != len(predicted_list) or not actual_list:
        raise ValueError("actual and predicted labels must be non-empty and equal length")
    labels = sorted(set(actual_list) | set(predicted_list))
    return {
        "n": len(actual_list),
        "labels": labels,
        "accuracy": float(accuracy_score(actual_list, predicted_list)),
        "precision_macro": float(precision_score(actual_list, predicted_list, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual_list, predicted_list, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(actual_list, predicted_list, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual_list, predicted_list, labels=labels).tolist(),
        "report": classification_report(actual_list, predicted_list, labels=labels, output_dict=True, zero_division=0),
    }


def _normalize(value: object, *, binary: bool, allowed: tuple[str, ...]) -> str:
    label = str(value).strip().lower()
    if binary:
        aliases = {"true": "yes", "1": "yes", "yes": "yes",
                   "false": "no", "0": "no", "no": "no"}
        if label not in aliases:
            raise ValueError(f"Invalid binary label: {value!r}")
        return aliases[label]
    if label not in allowed:
        raise ValueError(f"Invalid label {value!r}; expected one of {allowed}")
    return label


def evaluate_predictions(annotations: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate matched, eligible records. Blank annotations are omitted per task."""
    for name, frame in (("annotations", annotations), ("predictions", predictions)):
        if ("record_id" not in frame or frame["record_id"].isna().any()
                or frame["record_id"].astype(str).str.strip().eq("").any()):
            raise ValueError(f"{name} must contain non-empty record_id values")
        if frame["record_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate record_id values")
    merged = annotations.merge(
        predictions, on="record_id", how="left", suffixes=("_actual", "_predicted"),
        validate="one_to_one", indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] != "both", "record_id"]
    if not unmatched.empty:
        raise ValueError(f"Annotations without predictions: {unmatched.tolist()}")
    if merged.empty:
        raise ValueError("No annotated records to evaluate")
    if "is_before_cutoff" not in merged:
        raise ValueError("Predictions must include is_before_cutoff")
    eligible = merged["is_before_cutoff"].map(
        lambda value: _normalize(value, binary=True, allowed=()) == "yes"
    )
    if not eligible.all():
        raise ValueError("Evaluation includes records on or after the acquisition cutoff")

    tasks = {
        "climate_detection": ("is_climate_related", True, ()),
        "specificity": ("is_specific", True, ()),
        "risk_category": ("risk_category", False, RISK_CATEGORIES),
        "financial_relevance": ("financial_relevance", False, RELEVANCE_LEVELS),
    }
    results: dict[str, Any] = {}
    for task, (column, binary, allowed) in tasks.items():
        actual_col, predicted_col = f"{column}_actual", f"{column}_predicted"
        if actual_col not in merged or predicted_col not in merged:
            raise ValueError(f"Both inputs must contain the {column} column")
        labeled = merged[merged[actual_col].notna() & merged[actual_col].astype(str).str.strip().ne("")]
        if labeled.empty:
            continue
        if labeled[predicted_col].isna().any():
            raise ValueError(f"Missing prediction for {column}")
        actual = [_normalize(v, binary=binary, allowed=allowed) for v in labeled[actual_col]]
        predicted = [_normalize(v, binary=binary, allowed=allowed) for v in labeled[predicted_col]]
        results[task] = evaluate_labels(actual, predicted)
    if not results:
        raise ValueError("No labeled tasks to evaluate")
    return results
