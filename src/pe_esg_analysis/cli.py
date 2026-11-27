from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .classifiers import DisclosureClassifiers
from .climatebert import MODEL_ID, MODEL_REVISION, ClimateBertDetector
from .evaluation import evaluate_predictions
from .io import load_companies, load_disclosures, write_json, write_predictions, write_profiles
from .pipeline import ResearchPipeline, company_profiles, train_from_annotations


def run_analysis(args: argparse.Namespace) -> None:
    companies = load_companies(args.companies)
    disclosures = load_disclosures(args.disclosures)
    detector = ClimateBertDetector() if args.climate_detector == "climatebert" else None
    classifiers = DisclosureClassifiers(climate_detector=detector)
    predictions = ResearchPipeline(classifiers=classifiers).analyze(companies, disclosures)
    output_dir = Path(args.output_dir)
    write_predictions(output_dir / "predictions.csv", predictions)
    profiles = company_profiles(predictions)
    write_json(output_dir / "company_profiles.json", profiles)
    write_profiles(output_dir / "company_profiles.csv", profiles)
    write_json(output_dir / "run_metadata.json", {
        "climate_detector": args.climate_detector,
        "model_id": MODEL_ID if detector is not None else None,
        "model_revision": MODEL_REVISION if detector is not None else None,
        "other_classifiers": "heuristic",
    })
    print(f"Wrote {len(predictions)} predictions to {output_dir}")


def train_analysis(args: argparse.Namespace) -> None:
    companies = load_companies(args.companies)
    disclosures = load_disclosures(args.disclosures)
    annotations = pd.read_csv(args.annotations).fillna("").to_dict("records")
    disclosure_by_id = {record.record_id: record for record in disclosures}
    company_by_id = {company.company_id: company for company in companies}
    for row in annotations:
        record_id = str(row.get("record_id", ""))
        if record_id not in disclosure_by_id:
            raise ValueError(f"Training annotation has no matching disclosure: {record_id}")
        record = disclosure_by_id[record_id]
        if record.company_id not in company_by_id:
            raise ValueError(f"Unknown company_id: {record.company_id}")
        if record.document_date >= company_by_id[record.company_id].acquisition_announcement_date:
            raise ValueError(f"Training record is on or after acquisition cutoff: {record_id}")
    pipeline = train_from_annotations(ResearchPipeline(), disclosures, annotations)
    prediction_disclosures = (
        load_disclosures(args.prediction_disclosures) if args.prediction_disclosures else disclosures
    )
    if args.prediction_disclosures:
        training_ids = {str(row["record_id"]) for row in annotations}
        overlap = training_ids & {record.record_id for record in prediction_disclosures}
        if overlap:
            raise ValueError(f"Prediction records overlap training annotations: {sorted(overlap)}")
    else:
        print("Warning: predictions are in-sample; do not use them to report test performance.")
    predictions = pipeline.analyze(companies, prediction_disclosures)
    output_dir = Path(args.output_dir)
    write_predictions(output_dir / "predictions.csv", predictions)
    profiles = company_profiles(predictions)
    write_json(output_dir / "company_profiles.json", profiles)
    write_profiles(output_dir / "company_profiles.csv", profiles)
    write_json(output_dir / "run_metadata.json", {
        "climate_detector": "tfidf_logistic_regression",
        "prediction_scope": "held-out record IDs" if args.prediction_disclosures else "in-sample",
    })
    print(f"Wrote supervised predictions to {output_dir}")


def run_evaluation(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(args.predictions, dtype=str)
    annotations = pd.read_csv(args.annotations, dtype=str)
    results = evaluate_predictions(annotations, predictions)
    write_json(args.output, results)
    output = Path(args.output)
    summary = [
        {"task": task, **{key: metrics[key] for key in
                         ("n", "accuracy", "precision_macro", "recall_macro", "f1_macro")}}
        for task, metrics in results.items()
    ]
    classes = [
        {"task": task, "label": label, **metrics["report"][label]}
        for task, metrics in results.items()
        for label in metrics["labels"]
    ]
    pd.DataFrame(summary).to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    pd.DataFrame(classes).to_csv(output.with_name(output.stem + "_classes.csv"), index=False)
    print(f"Wrote evaluation results to {args.output}")


def run_extraction(args: argparse.Namespace) -> None:
    from .extraction import extract_corpus

    extract_corpus(args.manifest, args.ranges, args.pdf_dir, args.output, args.report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PE climate-risk disclosure research pipeline")
    subparsers = parser.add_subparsers(required=True)
    extract = subparsers.add_parser("extract", help="Extract dated 10-K PDF pages")
    extract.add_argument("--manifest", default="data/acquisition_manifest.csv")
    extract.add_argument("--ranges", default="data/extraction_ranges.csv")
    extract.add_argument("--pdf-dir", default="data/raw_documents")
    extract.add_argument("--output", default="data/processed_text/disclosures.csv")
    extract.add_argument("--report", default="data/processed_text/extraction_report.json")
    extract.set_defaults(func=run_extraction)
    run = subparsers.add_parser("run", help="Analyze extracted disclosure records")
    run.add_argument("--disclosures", required=True)
    run.add_argument("--companies", required=True)
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--climate-detector", choices=("climatebert", "heuristic"), default="climatebert")
    run.set_defaults(func=run_analysis)
    train = subparsers.add_parser("train", help="Fit baseline classifiers from manual annotations")
    train.add_argument("--disclosures", required=True)
    train.add_argument("--companies", required=True)
    train.add_argument("--annotations", required=True)
    train.add_argument("--prediction-disclosures", help="Separate held-out disclosures to predict")
    train.add_argument("--output-dir", default="outputs")
    train.set_defaults(func=train_analysis)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against manual annotations")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--output", default="outputs/evaluation.json")
    evaluate.set_defaults(func=run_evaluation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
