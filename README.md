# PE ESG Analysis

Reproducible research code for detecting and structuring climate-risk evidence in private-equity acquisition targets.

The pipeline intentionally does not include a dashboard. It produces auditable CSV/JSON outputs and publication-ready summary tables.

## Workflow

1. Prepare `data/company_metadata.csv` with `company_id`, `company_name`, `acquisition_announcement_date`, and optional `industry`.
2. Extract the locally downloaded PDFs (see `data/README.md` for current exclusions):

```powershell
python -m pip install -e ".[pdf,transformers,dev]"
python -m pe_esg_analysis.cli extract
```

The command writes `data/processed_text/disclosures.csv` and an extraction
report. Pages refer to 1-based physical PDF pages, not printed page numbers.
Review the report and extracted text before analysis. The command exits with an
error when any document or page is excluded or unreadable.

3. Run:
```powershell
python -m pip install -e ".[transformers,dev]"
python -m pe_esg_analysis.cli run `
  --disclosures data/processed_text/disclosures.csv `
  --companies data/company_metadata.csv `
  --output-dir outputs
```

The extractor uses PDF text blocks as approximate paragraphs, splitting blocks
over the model limit at word boundaries using the pinned ClimateBERT tokenizer.
PDF layout, tables, page breaks and OCR can require manual correction; the
result is not a validated annotation dataset. Files in `data/processed_text/`
are ignored by Git and should not be published without checking usage rights.
The default climate detector is the pretrained
[`climatebert/distilroberta-base-climate-detector`](https://huggingface.co/climatebert/distilroberta-base-climate-detector),
pinned to revision `2c3bc660d45a59e31b35f5d3e365ee4f59fdf76c`.
The first run downloads model files from Hugging Face; inference is local.
The detector is trained on English paragraphs, not sentences. Paragraphs longer than
512 tokens cause an error rather than silent truncation; split them before analysis.
`climate_confidence` is the predicted class probability, not a financial-risk score.
Use `--climate-detector heuristic` to run the original keyword baseline without
the optional transformer dependencies. Specificity, risk category and financial
relevance in `run` are still heuristic, not ClimateBERT outputs.

After manually labeling records, train the transparent supervised baseline:

```powershell
python -m pe_esg_analysis.cli train `
  --disclosures data/disclosures.csv `
  --companies data/company_metadata.csv `
  --annotations data/annotations.csv `
  --prediction-disclosures data/heldout_disclosures.csv `
  --output-dir outputs
```

Training and held-out records must not overlap. If `--prediction-disclosures`
is omitted, predictions are in-sample and unsuitable for reporting test performance.
For credible generalization, split by company or document before annotation/model
development; overlapping or near-duplicate passages can leak across record IDs.
The TF-IDF training command currently fits climate, specificity and risk models,
so its training annotations must supply at least two classes for each.

## Annotation format

For evaluation, create `data/annotations.csv` containing `record_id`, `is_climate_related`, `is_specific`, `risk_category`, and `financial_relevance`. Labels can be blank when only inference is required.

```powershell
python -m pe_esg_analysis.cli evaluate `
  --predictions outputs/predictions.csv `
  --annotations data/annotations.csv `
  --output outputs/evaluation.json
```

Evaluation writes JSON with per-class precision, recall, F1, support and confusion
matrices (rows are actual labels; columns are predictions), plus summary and
per-class CSV files. Macro scores average the labels observed in the evaluated
annotations or predictions. Blank labels are omitted independently for each task;
post-cutoff records and missing prediction IDs are rejected. Compare ClimateBERT
and TF-IDF using the same held-out annotated records and disclose the task-specific
sample counts. Evaluate climate detection separately from downstream predictions:
the latter are end-to-end results and depend on the climate gate.

The provided financial relevance rules are exploratory, not validated estimates of
financial impact. The evidence sentence is currently selected by length and requires
manual review. These outputs and model performance need validation against real
annotations before use in a research paper.
