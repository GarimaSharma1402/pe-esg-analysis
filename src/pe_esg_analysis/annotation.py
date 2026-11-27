"""Prepare independent company-level samples for human annotation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .io import load_companies
from .schemas import RELEVANCE_LEVELS, RISK_CATEGORIES

LABEL_COLUMNS = ("is_climate_related", "is_specific", "risk_category", "financial_relevance")
RECORD_COLUMNS = ("record_id", "company_id", "document_id", "document_date", "source", "page", "text")


def reviewed_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate annotation sheets before treating completed rows as ground truth."""
    if "annotation_status" not in frame:
        return frame
    missing = (set(LABEL_COLUMNS) | {"record_id", "notes"}) - set(frame.columns)
    if missing:
        raise ValueError(f"Annotation sheet missing columns: {sorted(missing)}")
    frame = frame.fillna("").copy()
    ids = frame["record_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Annotation sheet must have unique non-empty record IDs")
    statuses = frame["annotation_status"].astype(str).str.strip().str.lower()
    invalid = ~statuses.isin(("complete", "unusable"))
    if invalid.any():
        raise ValueError("Every annotation must have status complete or unusable")
    if (frame.loc[statuses == "unusable", "notes"].astype(str).str.strip() == "").any():
        raise ValueError("Unusable annotations require a reason in notes")
    completed = frame.loc[statuses == "complete"].copy()
    from .evaluation import _normalize

    for _, row in completed.iterrows():
        climate = _normalize(row["is_climate_related"], binary=True, allowed=())
        downstream = [str(row[column]).strip() for column in LABEL_COLUMNS[1:]]
        if climate == "no":
            if any(downstream):
                raise ValueError(f"Non-climate record {row['record_id']} must have blank downstream labels")
        else:
            _normalize(row["is_specific"], binary=True, allowed=())
            _normalize(row["risk_category"], binary=False, allowed=RISK_CATEGORIES)
            _normalize(row["financial_relevance"], binary=False, allowed=RELEVANCE_LEVELS)
    if completed.empty:
        raise ValueError("No completed annotations")
    return completed


def prepare_annotation_set(
    disclosures_path: str | Path,
    companies_path: str | Path,
    output_dir: str | Path,
    *,
    per_company: int = 50,
    test_companies: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    if per_company < 1:
        raise ValueError("per_company must be positive")
    frame = pd.read_csv(disclosures_path, dtype=str, keep_default_na=False)
    missing = set(RECORD_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing disclosure columns: {sorted(missing)}")
    if frame.empty or frame["record_id"].str.strip().eq("").any() or frame["record_id"].duplicated().any():
        raise ValueError("Disclosures must have unique non-empty record IDs")
    if frame[list(RECORD_COLUMNS)].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Disclosure provenance and text must be non-empty")
    companies = load_companies(companies_path)
    company_by_id = {company.company_id: company for company in companies}
    if len(company_by_id) != len(companies) or set(frame["company_id"]) != set(company_by_id):
        raise ValueError("Disclosure and company metadata must have matching unique company IDs")
    dates = pd.to_datetime(frame["document_date"], format="%Y-%m-%d", errors="raise").dt.date
    cutoffs = frame["company_id"].map(
        lambda company_id: company_by_id[company_id].acquisition_announcement_date
    )
    if (dates >= cutoffs).any():
        raise ValueError("Disclosures include records on or after acquisition cutoff")

    company_ids = sorted(company_by_id)
    if not 1 <= test_companies < len(company_ids):
        raise ValueError("test_companies must leave at least one company in each split")
    rng = random.Random(seed)
    rng.shuffle(company_ids)
    test_ids = set(company_ids[:test_companies])
    frame["split"] = frame["company_id"].map(lambda company_id: "test" if company_id in test_ids else "train")
    frame["_normalized"] = frame["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    split_counts = frame.groupby("_normalized")["split"].nunique()
    cross_split_texts = set(split_counts[split_counts > 1].index)
    eligible = frame[~frame["_normalized"].isin(cross_split_texts)]

    selected = []
    for company_id in sorted(company_by_id):
        pool = eligible[eligible["company_id"] == company_id].sort_values("record_id")
        if len(pool) < per_company:
            raise ValueError(f"{company_id} has only {len(pool)} eligible records; need {per_company}")
        selected.extend(rng.sample(pool["record_id"].tolist(), per_company))
    sample = frame[frame["record_id"].isin(selected)].sort_values(["company_id", "record_id"])
    output = Path(output_dir)
    paths = [output / f"{split}_{kind}.csv" for split in ("train", "test")
             for kind in ("disclosures", "annotations")]
    paths.append(output / "sampling_metadata.json")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite annotation files: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        part = sample[sample["split"] == split]
        disclosures = part[list(RECORD_COLUMNS)]
        disclosures.to_csv(output / f"{split}_disclosures.csv", index=False)
        annotations = disclosures.copy()
        for column in (*LABEL_COLUMNS, "annotation_status", "notes"):
            annotations[column] = ""
        annotations.to_csv(output / f"{split}_annotations.csv", index=False)
    metadata: dict[str, object] = {
        "seed": seed,
        "sampling": "uniform without replacement within each company after exact cross-split duplicate exclusion",
        "per_company": per_company,
        "train_companies": sorted(set(company_by_id) - test_ids),
        "test_companies": sorted(test_ids),
        "input_records": len(frame),
        "cross_split_duplicate_records_excluded": int(frame["_normalized"].isin(cross_split_texts).sum()),
        "selected_records": len(sample),
        "disclosures_sha256": hashlib.sha256(Path(disclosures_path).read_bytes()).hexdigest(),
    }
    (output / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
