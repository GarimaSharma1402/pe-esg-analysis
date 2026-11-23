from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import Company, Disclosure, Prediction


def _date(value: Any) -> date:
    return pd.to_datetime(value).date()


def load_companies(path: str | Path) -> list[Company]:
    frame = pd.read_csv(path)
    required = {"company_id", "company_name", "acquisition_announcement_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing company columns: {sorted(missing)}")
    return [
        Company(
            company_id=str(row.company_id),
            company_name=str(row.company_name),
            acquisition_announcement_date=_date(row.acquisition_announcement_date),
            industry="" if pd.isna(row.get("industry", "")) else str(row.get("industry", "")),
            esg_score=None if pd.isna(row.get("esg_score", None)) else float(row.esg_score),
        )
        for _, row in frame.iterrows()
    ]


def load_disclosures(path: str | Path) -> list[Disclosure]:
    frame = pd.read_csv(path)
    required = {"record_id", "company_id", "document_id", "document_date", "source", "page", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing disclosure columns: {sorted(missing)}")
    return [
        Disclosure(
            record_id=str(row.record_id),
            company_id=str(row.company_id),
            document_id=str(row.document_id),
            document_date=_date(row.document_date),
            source=str(row.source),
            page=str(row.page),
            text=str(row.text),
            section="" if pd.isna(row.get("section", "")) else str(row.get("section", "")),
        )
        for _, row in frame.iterrows()
    ]


def write_predictions(path: str | Path, predictions: list[Prediction]) -> None:
    frame = pd.DataFrame([prediction.as_dict() for prediction in predictions])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_profiles(path: str | Path, profiles: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(profiles)
    if "risk_category_counts" in frame.columns:
        frame["risk_category_counts"] = frame["risk_category_counts"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    if "relevance_counts" in frame.columns:
        frame["relevance_counts"] = frame["relevance_counts"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: str | Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
