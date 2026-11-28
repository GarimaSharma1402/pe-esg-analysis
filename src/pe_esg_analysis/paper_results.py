"""Produce descriptive outputs from human-reviewed disclosure annotations."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from .annotation import LABEL_COLUMNS, RECORD_COLUMNS, prepare_annotation_set, reviewed_annotations
from .io import load_companies, write_json
from .schemas import RELEVANCE_LEVELS, RISK_CATEGORIES


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reviewed_sheet(path: Path) -> pd.DataFrame:
    frame = _read(path)
    if "annotation_status" not in frame or set(RECORD_COLUMNS) - set(frame.columns):
        raise ValueError(f"{path} requires annotation status and complete provenance columns")
    reviewed_annotations(frame)
    return frame


def _cohorts(annotation_dir: Path) -> dict[str, pd.DataFrame]:
    train = _reviewed_sheet(annotation_dir / "train_annotations.csv")
    test = _reviewed_sheet(annotation_dir / "test_annotations.csv")
    challenge = _reviewed_sheet(annotation_dir / "suggested_test_challenge_candidates.csv")
    suggestions = pd.concat([
        _reviewed_sheet(annotation_dir / "suggested_training_candidates.csv"),
        _reviewed_sheet(annotation_dir / "suggested_training_candidates_round2.csv"),
    ], ignore_index=True)
    if suggestions["record_id"].duplicated().any():
        raise ValueError("Duplicate training suggestion IDs")
    merged = suggestions.merge(train, on="record_id", suffixes=("_suggestion", "_train"), how="left",
                               validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Training suggestions must be merged into the training sheet before reporting")
    for column in (*RECORD_COLUMNS[1:], *LABEL_COLUMNS, "annotation_status"):
        if (merged[f"{column}_suggestion"] != merged[f"{column}_train"]).any():
            raise ValueError(f"Training suggestion differs from merged annotation: {column}")
    candidate_ids = set(suggestions["record_id"])
    for name, sheet in (("train", train), ("test", test)):
        disclosures = _read(annotation_dir / f"{name}_disclosures.csv")
        if disclosures["record_id"].duplicated().any() or set(disclosures["record_id"]) != set(sheet["record_id"]):
            raise ValueError(f"{name} annotation and disclosure IDs do not match")
    return {
        "train_random": train[~train["record_id"].isin(candidate_ids)].copy(),
        "train_candidates": train[train["record_id"].isin(candidate_ids)].copy(),
        "test_random": test.copy(),
        "test_challenge": challenge.copy(),
    }


def _positive_figure(summary: pd.DataFrame) -> str:
    colors = {
        "physical": "#2673a6", "transition": "#d27b20", "regulatory": "#6d55a3",
        "environmental_liability": "#8f4735", "other": "#61756a",
    }
    labels = {
        "train_random": "Training random",
        "train_candidates": "Training candidates",
        "test_random": "Test random",
        "test_challenge": "Test challenge",
    }
    width, height, left, top = 900, 320, 210, 80
    scale = min(29, 600 / max(1, int(summary["positive"].max())))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Reviewed climate-positive disclosures by cohort</title>',
        '<desc id="description">Counts of reviewed positive records by risk category and selection cohort; '
        'not population prevalence.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222" font-size="15">',
        '<text x="20" y="28" font-size="19" font-weight="bold">Reviewed positive passages by cohort</text>',
        '<text x="20" y="51" font-size="13">Selected candidates and random samples are shown separately; not prevalence.</text>',
    ]
    for i, row in enumerate(summary.to_dict("records")):
        y = top + i * 39
        parts.append(f'<text x="20" y="{y + 16}">{escape(labels[row["cohort"]])}</text>')
        x = left
        for category in RISK_CATEGORIES:
            count = int(row[f"positive_{category}"])
            if count:
                w = count * scale
                parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="22" fill="{colors[category]}"/>')
                if w >= 25:
                    parts.append(f'<text x="{x + w / 2}" y="{y + 16}" text-anchor="middle" fill="white">{count}</text>')
                x += w
        parts.append(f'<text x="{x + 8}" y="{y + 16}">n={row["positive"]}</text>')
    for i, category in enumerate(RISK_CATEGORIES):
        x = 20 + i * 175
        label = category.replace("_", " ")
        parts.append(f'<rect x="{x}" y="260" width="14" height="14" fill="{colors[category]}"/>')
        parts.append(f'<text x="{x + 19}" y="272" font-size="12">{escape(label)}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def _evaluation_figure(metrics: pd.DataFrame) -> str:
    labels = {
        "random": "Random negatives correctly identified",
        "challenge": "Selected challenge positives detected",
    }
    colors = {"ClimateBERT": "#2673a6", "TF-IDF": "#d27b20"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" '
        'viewBox="0 0 900 300" role="img" aria-labelledby="title description">',
        '<title id="title">Exploratory held-out detector results</title>',
        '<desc id="description">Negative correct rate and selected-positive recall are different '
        'measures from different samples and are not overall model performance.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222" font-size="14">',
        '<text x="20" y="27" font-size="19" font-weight="bold">Exploratory held-out results</text>',
        '<text x="20" y="49" font-size="12">Different cohorts and outcomes; challenge sample is selected, not representative.</text>',
    ]
    for group_index, subset in enumerate(("random", "challenge")):
        base_y = 75 + group_index * 100
        parts.append(f'<text x="20" y="{base_y}">{escape(labels[subset])}</text>')
        for model_index, model in enumerate(("ClimateBERT", "TF-IDF")):
            match = metrics[(metrics["subset"] == subset) & (metrics["model"] == model)]
            if len(match) != 1:
                raise ValueError(f"Expected one comparison result for {subset}, {model}")
            row = match.iloc[0]
            numerator = int(row["tn"] if subset == "random" else row["tp"])
            denominator = int(row["negative_support"] if subset == "random" else row["positive_support"])
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                raise ValueError(f"Invalid support or correct count for {subset}, {model}")
            y = base_y + 12 + model_index * 28
            width = 400 * numerator / denominator
            parts.append(f'<text x="30" y="{y + 15}">{escape(model)}</text>')
            parts.append(f'<rect x="150" y="{y}" width="{width:.2f}" height="18" fill="{colors[model]}"/>')
            parts.append(f'<text x="565" y="{y + 15}">{numerator}/{denominator}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def generate_paper_results(
    *,
    companies_path: str | Path,
    disclosures_path: str | Path,
    annotation_dir: str | Path,
    comparison_path: str | Path,
    output_dir: str | Path,
) -> None:
    companies_path, disclosures_path = Path(companies_path), Path(disclosures_path)
    annotation_dir, comparison_path, output = Path(annotation_dir), Path(comparison_path), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results directory: {output}")
    companies = load_companies(companies_path)
    company_by_id = {company.company_id: company for company in companies}
    if len(company_by_id) != len(companies):
        raise ValueError("Company IDs must be unique")
    corpus = _read(disclosures_path)
    if set(RECORD_COLUMNS) - set(corpus.columns) or corpus["record_id"].duplicated().any():
        raise ValueError("Corpus requires unique IDs and complete provenance columns")
    corpus = corpus.set_index("record_id")
    cohorts = _cohorts(annotation_dir)
    split = json.loads((annotation_dir / "sampling_metadata.json").read_text(encoding="utf-8"))
    if split["disclosures_sha256"] != _sha256(disclosures_path):
        raise ValueError("Corpus differs from the original annotation sampling input")
    with tempfile.TemporaryDirectory() as temp_dir:
        replay = Path(temp_dir)
        replay_split = prepare_annotation_set(
            disclosures_path, companies_path, replay, per_company=split["per_company"],
            test_companies=len(split["test_companies"]), seed=split["seed"],
        )
        for side in ("train", "test"):
            if replay_split[f"{side}_companies"] != split[f"{side}_companies"]:
                raise ValueError(f"{side} company split does not match the recorded sampling seed")
            expected = set(_read(replay / f"{side}_disclosures.csv")["record_id"])
            if expected != set(cohorts[f"{side}_random"]["record_id"]):
                raise ValueError(f"{side} random cohort does not match the recorded sampling seed")
    all_rows = pd.concat([frame.assign(cohort=name) for name, frame in cohorts.items()], ignore_index=True)
    if all_rows["record_id"].duplicated().any():
        raise ValueError("An annotation ID occurs in multiple cohorts")
    if not set(all_rows["company_id"]).issubset(company_by_id):
        raise ValueError("Annotations contain an unknown company")
    for name, frame in cohorts.items():
        allowed = set(split["train_companies" if name.startswith("train_") else "test_companies"])
        if not set(frame["company_id"]).issubset(allowed):
            raise ValueError(f"Company split mismatch in {name}")
    for _, row in all_rows.iterrows():
        record_id = row["record_id"]
        if record_id not in corpus.index:
            raise ValueError(f"Annotation missing from corpus: {record_id}")
        for column in RECORD_COLUMNS[1:]:
            if row[column] != corpus.at[record_id, column]:
                raise ValueError(f"Annotation does not match corpus: {record_id} {column}")
        if pd.Timestamp(row["document_date"]).date() >= company_by_id[row["company_id"]].acquisition_announcement_date:
            raise ValueError(f"Post-acquisition annotation: {record_id}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison_inputs = {
        "companies": companies_path,
        "training_disclosures": annotation_dir / "train_disclosures.csv",
        "training_annotations": annotation_dir / "train_annotations.csv",
        "random_annotations": annotation_dir / "test_annotations.csv",
        "challenge_annotations": annotation_dir / "suggested_test_challenge_candidates.csv",
    }
    for key, path in comparison_inputs.items():
        if comparison["input_sha256"][key] != _sha256(path):
            raise ValueError(f"Comparison is stale for {key}; rerun compare-climate")
    metrics = pd.DataFrame(comparison["metrics"])

    summary = []
    profiles = []
    for name, frame in cohorts.items():
        complete = frame[frame["annotation_status"].str.strip().str.lower() == "complete"]
        positive = complete[complete["is_climate_related"].str.strip().str.lower().isin(["yes", "true", "1"])]
        counts = positive["risk_category"].str.strip().str.lower().value_counts()
        summary.append({
            "cohort": name, "total": len(frame), "reviewed": len(complete),
            "unusable": len(frame) - len(complete), "positive": len(positive),
            "positive_specific": int(positive["is_specific"].str.strip().str.lower().isin(["yes", "true", "1"]).sum()),
            **{f"positive_{category}": int(counts.get(category, 0)) for category in RISK_CATEGORIES},
            **{f"positive_relevance_{level}": int(
                (positive["financial_relevance"].str.strip().str.lower() == level).sum()
            ) for level in RELEVANCE_LEVELS},
        })
    summary_frame = pd.DataFrame(summary)
    for company in companies:
        row: dict[str, object] = {
            "company_id": company.company_id, "company_name": company.company_name,
            "acquisition_announcement_date": company.acquisition_announcement_date,
            "corpus_records_not_all_reviewed": int((corpus["company_id"] == company.company_id).sum()),
        }
        for name, frame in cohorts.items():
            subset = frame[frame["company_id"] == company.company_id]
            complete = subset[subset["annotation_status"].str.strip().str.lower() == "complete"]
            positive = complete[complete["is_climate_related"].str.strip().str.lower().isin(["yes", "true", "1"])]
            row[f"{name}_reviewed"] = len(complete)
            row[f"{name}_unusable"] = len(subset) - len(complete)
            row[f"{name}_positive"] = len(positive)
            row[f"{name}_positive_specific"] = int(
                positive["is_specific"].str.strip().str.lower().isin(["yes", "true", "1"]).sum()
            )
            for category in RISK_CATEGORIES:
                row[f"{name}_positive_{category}"] = int(
                    (positive["risk_category"].str.strip().str.lower() == category).sum()
                )
            for level in RELEVANCE_LEVELS:
                row[f"{name}_positive_relevance_{level}"] = int(
                    (positive["financial_relevance"].str.strip().str.lower() == level).sum()
                )
        profiles.append(row)
    complete = all_rows[all_rows["annotation_status"].str.strip().str.lower() == "complete"]
    evidence = complete[complete["is_climate_related"].str.strip().str.lower().isin(["yes", "true", "1"])]

    positive_figure = _positive_figure(summary_frame)
    evaluation_figure = _evaluation_figure(metrics)
    output.mkdir(parents=True)
    pd.DataFrame(profiles).to_csv(output / "reviewed_company_profiles.csv", index=False)
    summary_frame.to_csv(output / "annotation_cohort_summary.csv", index=False)
    evidence[["cohort", *RECORD_COLUMNS, *LABEL_COLUMNS, "notes"]].to_csv(
        output / "reviewed_positive_evidence.csv", index=False
    )
    metrics.to_csv(output / "climate_detection_evaluation.csv", index=False)
    (output / "reviewed_positive_categories.svg").write_text(positive_figure, encoding="utf-8")
    (output / "climate_detection_evaluation.svg").write_text(evaluation_figure, encoding="utf-8")
    write_json(output / "results_metadata.json", {
        "basis": "Human-reviewed annotations only; no full-corpus model analysis.",
        "companies": len(companies),
        "annotation_records": len(all_rows),
        "reviewed_positive_records": len(evidence),
        "esg_comparison": "Not performed: the metadata lacks dated, source-verified pre-acquisition ESG scores.",
        "limitations": [
            "Random samples and targeted candidates must not be combined as prevalence estimates.",
            "No reviewed positive passage does not mean a company has no climate risk or disclosure.",
            "The challenge was selected and previously inspected; it is not a pristine independent test.",
            "Model evaluation is underpowered, with no positives in the random held-out sample.",
            "Review source PDF context, extraction quality, dates and evidence before publication.",
        ],
        "input_sha256": {
            "companies": _sha256(companies_path), "corpus": _sha256(disclosures_path),
            "comparison": _sha256(comparison_path),
            **{name: _sha256(annotation_dir / name) for name in (
                "sampling_metadata.json", "train_disclosures.csv", "test_disclosures.csv",
                "train_annotations.csv", "test_annotations.csv", "suggested_training_candidates.csv",
                "suggested_training_candidates_round2.csv", "suggested_test_challenge_candidates.csv",
            )},
        },
    })
