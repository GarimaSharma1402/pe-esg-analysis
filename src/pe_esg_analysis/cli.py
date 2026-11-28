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
    from .annotation import reviewed_annotations

    companies = load_companies(args.companies)
    disclosures = load_disclosures(args.disclosures)
    annotations = reviewed_annotations(pd.read_csv(args.annotations, dtype=str)).fillna("").to_dict("records")
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
    from .annotation import reviewed_annotations

    predictions = pd.read_csv(args.predictions, dtype=str)
    annotations = reviewed_annotations(pd.read_csv(args.annotations, dtype=str))
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


def prepare_annotations(args: argparse.Namespace) -> None:
    from .annotation import prepare_annotation_set

    metadata = prepare_annotation_set(
        args.disclosures, args.companies, args.output_dir,
        per_company=args.per_company, test_companies=args.test_companies, seed=args.seed,
    )
    print(f"Prepared {metadata['selected_records']} blank annotation records in {args.output_dir}")


def compare_climate(args: argparse.Namespace) -> None:
    from .comparison import compare_climate_detectors

    compare_climate_detectors(
        companies_path=args.companies,
        training_disclosures_path=args.training_disclosures,
        training_annotations_path=args.training_annotations,
        random_annotations_path=args.random_annotations,
        challenge_annotations_path=args.challenge_annotations,
        output_dir=args.output_dir,
    )
    print(f"Wrote separate climate detector comparisons to {args.output_dir}")


def paper_results(args: argparse.Namespace) -> None:
    from .paper_results import generate_paper_results

    generate_paper_results(
        companies_path=args.companies,
        disclosures_path=args.disclosures,
        annotation_dir=args.annotation_dir,
        comparison_path=args.comparison,
        output_dir=args.output_dir,
    )
    print(f"Wrote reviewed-sample results to {args.output_dir}")


def run_full_analysis(args: argparse.Namespace) -> None:
    from .full_corpus import run_full_corpus

    run_full_corpus(args.companies, args.disclosures, args.output_dir, args.batch_size)
    print(f"Wrote full-corpus screening to {args.output_dir}")


def reviewed_screening_results(args: argparse.Namespace) -> None:
    from .reviewed_results import generate_reviewed_results

    generate_reviewed_results(
        companies_path=args.companies, predictions_path=args.predictions,
        decisions_path=args.decisions, source_checks_path=args.source_checks,
        annotation_dir=args.annotation_dir, comparison_path=args.comparison,
        manifest_path=args.manifest, extraction_report_path=args.extraction_report,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote reviewed screening results to {args.output_dir}")


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
    annotate = subparsers.add_parser("prepare-annotations", help="Sample records for human annotation")
    annotate.add_argument("--disclosures", default="data/processed_text/disclosures.csv")
    annotate.add_argument("--companies", default="data/company_metadata.csv")
    annotate.add_argument("--output-dir", default="data/processed_text/annotations")
    annotate.add_argument("--per-company", type=int, default=50)
    annotate.add_argument("--test-companies", type=int, default=4)
    annotate.add_argument("--seed", type=int, default=42)
    annotate.set_defaults(func=prepare_annotations)
    compare = subparsers.add_parser("compare-climate", help="Compare climate detectors on reviewed cohorts")
    compare.add_argument("--companies", default="data/company_metadata.csv")
    compare.add_argument("--training-disclosures", default="data/processed_text/annotations/train_disclosures.csv")
    compare.add_argument("--training-annotations", default="data/processed_text/annotations/train_annotations.csv")
    compare.add_argument("--random-annotations", default="data/processed_text/annotations/test_annotations.csv")
    compare.add_argument("--challenge-annotations", default="data/processed_text/annotations/suggested_test_challenge_candidates.csv")
    compare.add_argument("--output-dir", default="outputs/climate_comparison")
    compare.set_defaults(func=compare_climate)
    paper = subparsers.add_parser("paper-results", help="Summarize reviewed annotations for paper preparation")
    paper.add_argument("--companies", default="data/company_metadata.csv")
    paper.add_argument("--disclosures", default="data/processed_text/disclosures.csv")
    paper.add_argument("--annotation-dir", default="data/processed_text/annotations")
    paper.add_argument("--comparison", default="outputs/climate_comparison/comparison.json")
    paper.add_argument("--output-dir", default="outputs/paper_results")
    paper.set_defaults(func=paper_results)
    full = subparsers.add_parser("run-full", help="Run batched ClimateBERT screening by company")
    full.add_argument("--companies", default="data/company_metadata.csv")
    full.add_argument("--disclosures", default="data/processed_text/disclosures.csv")
    full.add_argument("--output-dir", default="outputs/full_corpus_climatebert")
    full.add_argument("--batch-size", type=int, default=32)
    full.set_defaults(func=run_full_analysis)
    screening = subparsers.add_parser("reviewed-screening-results", help="Assemble reviewed model-flag results")
    screening.add_argument("--companies", default="data/company_metadata.csv")
    screening.add_argument("--predictions", default="outputs/full_corpus_climatebert/predictions.csv")
    screening.add_argument("--decisions", default="outputs/full_corpus_climatebert/review_decisions.csv")
    screening.add_argument("--source-checks", default="outputs/full_corpus_climatebert/all_flagged_source_checks.csv")
    screening.add_argument("--annotation-dir", default="data/processed_text/annotations")
    screening.add_argument("--comparison", default="data/processed_text/step4_reproducibility_check/comparison.json")
    screening.add_argument("--manifest", default="data/acquisition_manifest.csv")
    screening.add_argument("--extraction-report", default="data/processed_text/extraction_report.json")
    screening.add_argument("--pdf-dir", default="data/raw_documents")
    screening.add_argument("--output-dir", default="outputs/paper_results_screening")
    screening.set_defaults(func=reviewed_screening_results)
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
