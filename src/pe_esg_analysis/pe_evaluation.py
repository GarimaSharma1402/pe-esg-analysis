"""Prepare and evaluate a blinded, stratified PE passage review."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .annotation import RECORD_COLUMNS, reviewed_annotations
from .io import load_disclosures, write_json
from .pipeline import ResearchPipeline, train_from_annotations
from .preprocessing import clean_text


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _allocate(populations: dict[str, int], budget: int) -> dict[str, int]:
    """Census model flags where possible and reserve at least one other record."""
    positive = populations["model_flagged"]
    negative = populations["neither_flagged"]
    if min(positive, negative) < 1:
        raise ValueError("Both model-flag strata must be non-empty")
    if budget < 2 or budget > positive + negative:
        raise ValueError("Review budget must be between 2 and the eligible population size")
    flagged = min(positive, budget - 1)
    remaining = budget - flagged
    if remaining > negative:
        remaining = negative
        flagged = budget - remaining
    return {"model_flagged": flagged, "neither_flagged": remaining}


def prepare_pe_evaluation(
    *, disclosures_path: str | Path, predictions_path: str | Path,
    run_metadata_path: str | Path, annotation_dir: str | Path,
    comparison_path: str | Path, paired_path: str | Path,
    output_dir: str | Path, budget: int = 400, seed: int = 42,
) -> None:
    disclosures_path, predictions_path, run_metadata_path, annotation_dir = map(
        Path, (disclosures_path, predictions_path, run_metadata_path, annotation_dir)
    )
    comparison_path, paired_path, output = map(Path, (comparison_path, paired_path, output_dir))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review directory: {output}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    sampling = json.loads((annotation_dir / "sampling_metadata.json").read_text(encoding="utf-8"))
    run = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    for name, path in {
        "training_disclosures": annotation_dir / "train_disclosures.csv",
        "training_annotations": annotation_dir / "train_annotations.csv",
        "random_annotations": annotation_dir / "test_annotations.csv",
        "challenge_annotations": annotation_dir / "suggested_test_challenge_candidates.csv",
    }.items():
        if comparison["input_sha256"][name] != _hash(path):
            raise ValueError(f"Training or evaluation input differs from prior comparison: {name}")
    if sampling["disclosures_sha256"] != _hash(disclosures_path) or run["input_sha256"]["disclosures"] != _hash(disclosures_path):
        raise ValueError("Disclosures differ from the original split or full-corpus run")
    if run["status"] != "complete" or run["model_revision"] != comparison["climatebert_revision"]:
        raise ValueError("Full-corpus run is incomplete or uses a different ClimateBERT revision")

    disclosures, predictions = _read(disclosures_path), _read(predictions_path)
    if disclosures["record_id"].duplicated().any() or predictions["record_id"].duplicated().any():
        raise ValueError("Disclosure and prediction IDs must be unique")
    if set(disclosures["record_id"]) != set(predictions["record_id"]):
        raise ValueError("Full-corpus predictions do not cover the disclosures")
    if not predictions["is_climate_related"].isin(("True", "False")).all():
        raise ValueError("Invalid ClimateBERT prediction")
    training_ids = set(sampling["train_companies"])
    test_ids = set(sampling["test_companies"])
    if training_ids & test_ids or set(disclosures["company_id"]) != training_ids | test_ids:
        raise ValueError("Invalid company split")
    disclosures["_normalized"] = disclosures["text"].map(_normalize_text)
    prior = pd.concat([
        _read(annotation_dir / "test_annotations.csv"),
        _read(annotation_dir / "suggested_test_challenge_candidates.csv"),
    ], ignore_index=True)
    excluded = set(disclosures.loc[disclosures["company_id"].isin(training_ids), "_normalized"])
    excluded.update(prior["text"].map(_normalize_text))
    heldout = disclosures[disclosures["company_id"].isin(test_ids)].copy()
    scored = heldout.merge(
        predictions[["record_id", "company_id", "is_climate_related", "is_before_cutoff", "text"]],
        on="record_id", validate="one_to_one", suffixes=("", "_prediction"),
    )
    if not scored["company_id"].eq(scored["company_id_prediction"]).all() or not scored["is_before_cutoff"].eq("True").all():
        raise ValueError("Held-out prediction provenance or cutoff mismatch")
    if not scored["text"].map(clean_text).eq(scored["text_prediction"]).all():
        raise ValueError("Held-out prediction text differs from disclosure")

    training = reviewed_annotations(_read(annotation_dir / "train_annotations.csv")).to_dict("records")
    baseline = train_from_annotations(
        ResearchPipeline(), load_disclosures(annotation_dir / "train_disclosures.csv"), training
    ).classifiers.climate_model
    if baseline.model is None:
        raise RuntimeError("TF-IDF baseline was not fitted")
    probabilities = baseline.model.predict_proba(scored["text"].map(clean_text).tolist())
    scored["tfidf"] = [
        baseline.classes[index].casefold() in {"yes", "true", "1", "climate"}
        for index in np.argmax(probabilities, axis=1)
    ]
    scored["climatebert"] = scored["is_climate_related"].eq("True")
    paired = _read(paired_path)
    if paired["record_id"].duplicated().any():
        raise ValueError("Prior paired prediction IDs must be unique")
    check = paired.merge(scored[["record_id", "climatebert", "tfidf"]], on="record_id", validate="one_to_one")
    if len(check) != len(paired) or any(
        not check[model].eq(check[column].eq("yes")).all()
        for model, column in (("climatebert", "ClimateBERT"), ("tfidf", "TF-IDF"))
    ):
        raise ValueError("Model predictions differ from the prior held-out comparison")

    eligible = scored[~scored["_normalized"].isin(excluded)].copy()
    eligible["stratum"] = np.where(
        eligible["climatebert"] | eligible["tfidf"], "model_flagged", "neither_flagged"
    )
    populations = {name: int(eligible["stratum"].eq(name).sum()) for name in ("model_flagged", "neither_flagged")}
    allocation = _allocate(populations, budget)
    rng = random.Random(seed)
    selected_ids: list[str] = []
    for stratum, count in allocation.items():
        ids = sorted(eligible.loc[eligible["stratum"].eq(stratum), "record_id"].tolist())
        selected_ids.extend(rng.sample(ids, count))
    rng.shuffle(selected_ids)
    selected = eligible.set_index("record_id").loc[selected_ids].reset_index()
    reference = selected[list(RECORD_COLUMNS)].copy()
    labels = reference.copy()
    labels["is_climate_related"] = ""
    labels["annotation_status"] = ""
    labels["notes"] = ""
    frozen = selected[["record_id", "stratum", "climatebert", "tfidf"]].copy()
    frozen["population_size"] = frozen["stratum"].map(populations)
    frozen["sample_size"] = frozen["stratum"].map(allocation)
    output.mkdir(parents=True)
    labels.to_csv(output / "review_sheet.csv", index=False)
    reference.to_csv(output / "sample_reference.csv", index=False)
    frozen.to_csv(output / "frozen_predictions_and_design.csv", index=False)
    write_json(output / "sampling_metadata.json", {
        "seed": seed, "heldout_companies": sorted(test_ids),
        "heldout_records": len(heldout), "eligible_records": len(eligible),
        "stratum_populations": populations, "stratum_sample_sizes": allocation,
        "design": "Simple random sampling without replacement within either-model-positive and neither-positive strata.",
        "exclusions": "Exact normalized text appearing in any training-company record or previously annotated test/challenge record.",
        "label_task": "Climate relevance only, using the existing weather-hazard screening codebook.",
        "limitations": [
            "These four companies and prior model outputs were previously analyzed; this is not a fresh company-level test.",
            "ClimateBERT-flagged passages have already been examined by an AI assistant; the sheet hides model outputs but prior exposure remains possible.",
            "Counts refer only to eligible extracted records from these four companies, not all PE targets or all records in the corpus.",
            "Model training and thresholds must remain unchanged after reviewing the sample.",
            "Low positive support and only four companies may make estimated metrics highly uncertain.",
        ],
        "input_sha256": {
            "disclosures": _hash(disclosures_path), "full_predictions": _hash(predictions_path),
            "comparison": _hash(comparison_path), "paired_predictions": _hash(paired_path),
            "train_disclosures": _hash(annotation_dir / "train_disclosures.csv"),
            "train_annotations": _hash(annotation_dir / "train_annotations.csv"),
        },
        "sklearn_version": sklearn.__version__,
        "reference_sha256": _hash(output / "sample_reference.csv"),
        "frozen_design_sha256": _hash(output / "frozen_predictions_and_design.csv"),
    })


def _metrics(frame: pd.DataFrame, model: str) -> dict[str, int | float | None]:
    predicted = frame[model].eq("True")
    actual = frame["is_climate_related"].eq("yes")
    weight = frame["population_size"].astype(float) / frame["sample_size"].astype(float)
    tp = float(weight[actual & predicted].sum())
    fn = float(weight[actual & ~predicted].sum())
    fp = float(weight[~actual & predicted].sum())
    tn = float(weight[~actual & ~predicted].sum())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    positive_f1 = ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None
    negative_f1 = ratio(2 * tn, 2 * tn + fp + fn) if tn + fp else None
    return {
        "reviewed_records": len(frame),
        "reviewed_positives": int(actual.sum()), "reviewed_negatives": int((~actual).sum()),
        "weighted_tp": tp, "weighted_fn": fn, "weighted_fp": fp, "weighted_tn": tn,
        "accuracy": ratio(tp + tn, tp + fn + fp + tn),
        "positive_precision": precision, "positive_recall": recall, "positive_f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2 if positive_f1 is not None and negative_f1 is not None else None,
    }


def evaluate_pe_review(review_dir: str | Path, output_dir: str | Path) -> None:
    review_dir, output = Path(review_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output}")
    meta = json.loads((review_dir / "sampling_metadata.json").read_text(encoding="utf-8"))
    reference_path, frozen_path = review_dir / "sample_reference.csv", review_dir / "frozen_predictions_and_design.csv"
    if _hash(reference_path) != meta["reference_sha256"] or _hash(frozen_path) != meta["frozen_design_sha256"]:
        raise ValueError("Frozen review inputs have changed")
    reference, frozen, labels = map(_read, (reference_path, frozen_path, review_dir / "review_sheet.csv"))
    if (
        reference["record_id"].duplicated().any()
        or frozen["record_id"].duplicated().any()
        or set(frozen["record_id"]) != set(reference["record_id"])
    ):
        raise ValueError("Frozen design must cover exactly the sampled record IDs")
    if not frozen[["climatebert", "tfidf"]].isin(("True", "False")).all().all():
        raise ValueError("Invalid frozen model prediction")
    expected_strata = np.where(
        frozen["climatebert"].eq("True") | frozen["tfidf"].eq("True"), "model_flagged", "neither_flagged"
    )
    if not frozen["stratum"].eq(expected_strata).all():
        raise ValueError("Frozen stratum differs from model predictions")
    for stratum, population in meta["stratum_populations"].items():
        rows = frozen[frozen["stratum"].eq(stratum)]
        sample_size = meta["stratum_sample_sizes"][stratum]
        if (
            len(rows) != sample_size or not rows["population_size"].eq(str(population)).all()
            or not rows["sample_size"].eq(str(sample_size)).all() or not 1 <= sample_size <= population
        ):
            raise ValueError(f"Frozen sampling counts are inconsistent: {stratum}")
    if set(frozen["stratum"]) != set(meta["stratum_populations"]):
        raise ValueError("Frozen strata differ from sampling metadata")
    if labels["record_id"].duplicated().any() or set(labels["record_id"]) != set(reference["record_id"]):
        raise ValueError("Review sheet must contain exactly the sampled record IDs")
    label_reference = labels[list(RECORD_COLUMNS)].set_index("record_id").sort_index()
    if not label_reference.equals(reference.set_index("record_id").sort_index()):
        raise ValueError("Review sheet provenance or source text has changed")
    labels["annotation_status"] = labels["annotation_status"].str.strip().str.lower()
    labels["is_climate_related"] = labels["is_climate_related"].str.strip().str.lower()
    if not labels["annotation_status"].isin(("complete", "unusable")).all():
        raise ValueError("Every record must be complete or unusable")
    if not labels.loc[labels["annotation_status"].eq("complete"), "is_climate_related"].isin(("yes", "no")).all():
        raise ValueError("Completed records require a yes/no climate label")
    unusable = labels["annotation_status"].eq("unusable")
    if labels.loc[unusable, "notes"].str.strip().eq("").any() or labels.loc[unusable, "is_climate_related"].ne("").any():
        raise ValueError("Unusable records require a reason and a blank climate label")
    merged = frozen.merge(labels[["record_id", "annotation_status", "is_climate_related"]], on="record_id", validate="one_to_one")
    usable = merged[merged["annotation_status"].eq("complete")].copy()
    if usable.empty or set(usable["stratum"]) != set(frozen["stratum"]):
        raise ValueError("Each sampling stratum needs at least one usable reviewed record")
    results = [{"model": label, **_metrics(usable, column)} for label, column in (
        ("ClimateBERT", "climatebert"), ("TF-IDF", "tfidf"),
    )]
    output.mkdir(parents=True)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    write_json(output / "evaluation.json", {
        "task": "Climate relevance only", "metrics": results,
        "reviewed_records": len(usable), "unusable_records": int(unusable.sum()),
        "unusable_by_stratum": merged.loc[merged["annotation_status"].eq("unusable"), "stratum"].value_counts().to_dict(),
        "estimation": "Inverse-probability weighted confusion counts using original stratum populations and sample sizes; unusable records are excluded.",
        "limitations": meta["limitations"] + [
            "Metrics estimate performance on usable eligible passages, not model performance across all records or companies.",
            "Usability is known only for sampled records; excluded unusable records are not assumed negative.",
            "No confidence intervals are calculated; do not treat point estimates as precise.",
            "Precision and F1 depend on the target population and annotation codebook.",
        ],
        "review_sha256": _hash(review_dir / "review_sheet.csv"),
        "sampling_metadata_sha256": _hash(review_dir / "sampling_metadata.json"),
    })
