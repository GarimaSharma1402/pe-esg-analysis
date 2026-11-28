# Acquisition sample and reports

This convenience sample contains 20 public-company targets with announced
acquisitions in 2015–2024 and sources reporting that the transactions completed.
It is heavily concentrated in software companies and is not representative of
all PE targets. Company inclusion is based on available reports and sources.

* `company_metadata.csv` provides the company metadata expected by the pipeline.
* `acquisition_manifest.csv` records the acquirer, announcement date and source,
  fiscal report year, 10-K filing date and source, report download URL and filename.
* `acquisition_completions.csv` records sources for transaction completion.
* `raw_documents/` contains the downloaded reports. This folder is ignored by Git.

The SEC returned HTTP 403 in this environment. The PDFs were downloaded from
AnnualReports.com or the alternative URL in the manifest, not directly from
EDGAR. Filing dates were cross-checked against the cited SEC or filing-index
sources and are strictly before the announcement dates. The sources identify
the filing dates of the included 10-Ks, not necessarily publication dates of
every page of the surrounding annual-report PDF.

`extraction_ranges.csv` records the selected inclusive physical PDF page ranges
for the 10-Ks; shareholder letters, proxy statements and separate exhibits
are excluded rather than being assigned a 10-K filing date. The selected pages
and their filing dates still require human review. Two original files, Alteryx
and Zendesk, were damaged; verified fresh downloads are referenced in the
manifest, while the original local files remain untouched.

Run `python -m pe_esg_analysis.cli extract` after installing the `pdf` and
`transformers` extras. The output and a per-page report are written under
`processed_text/`, which is ignored by Git. The report includes the PDF SHA-256,
physical page number, extraction status, and record counts. Unreadable/blank
pages and excluded companies make the command exit nonzero, even though it
writes the partial output for inspection. PDF blocks are only approximations of
paragraphs; tables, headers, continuation across pages and reading order need
manual quality checks. Records exceeding the model limit are split without
truncation, so one paragraph may become multiple records.

The original PowerSchool 2022 PDF had garbled text. The manifest now references
a readable alternative copy of the same 10-K (accession 0001835681-23-000012).
The original Model N 2022 PDF had a garbled cover; the manifest instead uses
its readable 2021 10-K (accession 0001118417-21-000166), filed on 2021-11-19.
This changes Model N's report year and may change its observed disclosures.
The selected ranges and extracted text still require human quality review.

For Anaplan and Medallia, the archive's URL label says 2020, but the PDFs
contain fiscal-year 2021 10-Ks; the filenames and manifest reflect their
actual report periods. This corpus is not yet a labeled dataset.

## Manual annotation

After reviewing extraction quality, run:

```powershell
python -m pe_esg_analysis.cli prepare-annotations
```

The default seed is 42. The command assigns companies to splits before sampling:
16 train companies and 4 test companies, then 50 records per company without
replacement. It excludes text that is identical after whitespace normalization
and case-folding across the two company pools, to prevent exact-text leakage.
The seed, company assignments, exclusion count and input SHA-256 are in
`processed_text/annotations/sampling_metadata.json`. Four CSVs contain the
train/test disclosure inputs and corresponding blank annotation sheets.
Files are local and Git-ignored; the command refuses to overwrite them.

AI-assisted candidates are kept separate from the random sample until human
review. Reviewed training candidates may be added to the training annotation
and disclosure sheets with their provenance noted. Keep any climate-enriched
held-out supplement separate from the random test sample, and report its
results only as challenge-set results, not as random-sample performance or
prevalence. AI-proposed labels are not ground truth.

This 1,000-record set is a starting point, not an industry-standard or
statistically justified sample size. The test set contains only four companies.
Equal numbers per company and duplicate exclusion mean unweighted results do
not estimate prevalence or performance across all 30,983 records. Random
sampling may yield very few climate examples or category classes; report class
support and uncertainty, and do not claim reliable category F1 when support is
insufficient. Similar but non-identical text may still cross splits. Do not
tune labels, thresholds or models on the held-out companies. The extraction
report still flags pages for review and the input text blocks are approximate.

### Proposed annotation codebook

Annotate the evidence in each record, checking its source and physical PDF page
when context or extraction quality is unclear. These rules are a starting
protocol to review and freeze before annotating the held-out test set:

1. `is_climate_related`: `yes` if the text explicitly discusses climate change,
   emissions or a climate-related transition or regulatory issue. Also label
   disclosed weather hazards (for example, floods, hurricanes, severe storms,
   tornadoes, droughts, heatwaves or wildfires) as physical-risk evidence even
   when the text does not explicitly attribute them to climate change. This is
   an operational screening rule for weather-hazard exposure, not a claim that
   climate change caused a particular event. Do not qualify non-weather
   hazards such as earthquakes on their own, or generic environmental,
   governance or legal statements.
2. For `yes` records, `is_specific` is `yes` when the text gives concrete
   company-relevant exposure, activity, asset, location, measure or consequence;
   a general aspiration or generic statement is `no`.
3. For `yes` records, choose one primary `risk_category`: `physical` for climate
   hazards; `transition` for shifts in markets, technology or energy; `regulatory`
   for climate-related laws or compliance; `environmental_liability` for an
   explicitly climate-linked cleanup, contamination or liability; or `other`
   for climate-related text without an identifiable risk category. Record
   secondary categories or ambiguity in `notes`. The climate gate means
   environmental liabilities without an explicit climate connection are not
   covered by downstream risk evaluation.
4. For `yes` records, `financial_relevance` is `low` for general statements
   without an identifiable financial implication, `medium` for an identifiable
   potential cost, loss or financial exposure without a clear material effect,
   and `high` for a disclosed significant financial effect or explicit material
   exposure. Do not invent amounts or assume impacts not supported by the text.
   These are proposed qualitative labels, not audited estimates of impact.

For `no` records leave all three downstream labels blank. Mark
`annotation_status` as `complete` when applicable labels are finished, or
`unusable` with a reason in `notes` for an unreadable fragment or insufficient
context; do not silently label unusable text as negative. Every row must have a
status before the provided train/evaluate commands accept the sheet. Two
independent annotators and adjudication of disagreements, at least on a subset,
are recommended; record who annotated and document any protocol changes.
Do not treat AI suggestions as independent human ground truth.

## Reviewed-sample paper outputs

After running `compare-climate`, generate local tables and SVG figures with:

```powershell
python -m pe_esg_analysis.cli paper-results --comparison outputs/climate_comparison/comparison.json --output-dir outputs/paper_results
```

The command requires the reviewed training and test sheets, both reviewed
training-candidate sheets, the separate challenge sheet, original sampling
metadata, and the original disclosure corpus. It reconstructs the seeded random
sample and checks corpus provenance, company split, pre-announcement dates,
merged candidate labels and the comparison input hashes. It refuses to overwrite
an existing output directory. Set `--annotation-dir`, `--disclosures`,
`--companies`, or `--comparison` if the inputs are elsewhere.

`reviewed_company_profiles.csv` reports reviewed, unusable and positive passage
counts, plus specificity, category and qualitative financial-relevance counts,
separately for random and selected cohorts. `reviewed_positive_evidence.csv`
retains each reviewed positive passage, label and document/page provenance.
`annotation_cohort_summary.csv` and `reviewed_positive_categories.svg` describe
the four cohorts; `climate_detection_evaluation.csv` and its SVG show the prior
climate-detection comparison. `results_metadata.json` records hashes and
limitations. Outputs include disclosure text and remain local and Git-ignored;
check publication and reuse rights before sharing.

These are profiles **of the reviewed sample**, not full-document or full-corpus
company risk assessments. A zero count is not evidence that a company has no
climate risk. Random sampling follows company stratification and cross-split
exact-duplicate exclusion; candidate/challenge cases were selected and cannot
be pooled with random cases to estimate prevalence. AI-suggested cases were
human-reviewed, but not independently annotated; the challenge was already
inspected. Test cohorts are single-class, so the comparison does not provide
reliable overall precision, F1 or generalization. Weather hazards are a
screening convention, not proof of climate causation. Financial labels are
qualitative, not impact estimates. Confirm PDF context, extraction quality,
document dating and annotation decisions before publication.

No ESG-score comparison is produced: scores in the current company metadata
are missing and that schema does not record score provider, vintage or
publication date. Any future comparison requires actual scores from a
consistent provider, documented scale and direction, dates strictly before
each announcement, source provenance and appropriate usage rights.

## Full-corpus ClimateBERT screening

With the optional `transformers` dependencies and pinned model available
locally, run:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pe_esg_analysis.cli run-full --output-dir outputs/full_corpus_climatebert
```

The command refuses to overwrite an existing output directory. It verifies
company IDs, unique record IDs and pre-announcement dates, batches inputs
without truncation, and reuses predictions for identical cleaned text. It
writes per-company prediction files as they finish; the combined predictions,
company profiles and `status: complete` metadata appear only after every
company finishes. A partially completed run must not be used as full-corpus
results. The process does not automatically resume a partial run.

Model-positive passages, categories and financial-relevance labels are
**unreviewed screening outputs**, not verified company risks or financial
impact estimates. Aggregated counts depend on PDF extraction, passage splitting
and report coverage; they are not comparable risk scores. ClimateBERT may
produce false positives and false negatives, including weather passages. The
other classifiers and evidence-sentence selector remain exploratory rules.
Inspect complete passages and their cited PDF pages before using results as
evidence; do not assume the selected sentence carries the context of a record.

## Exploratory full-corpus paper package

After screening, the local `review_decisions.csv` and
`all_flagged_source_checks.csv` record the assistant's review and PDF-block
checks of all model flags, plus the previously user-reviewed positives missed
by the model. These review files are inputs, not automatically generated human
annotations. Assemble the package without overwriting an existing directory:

```powershell
python -m pe_esg_analysis.cli reviewed-screening-results --output-dir outputs/paper_results_screening_final
```

The command checks review coverage, preserves user labels, checks manifest
dates and source identifiers, verifies current PDF hashes against the
extraction report, and checks comparison input hashes. It relies on the
recorded block-check ledger rather than repeating PDF-block extraction. The
package contains `company_profiles.csv`, `reviewed_evidence.csv`,
`review_outcome_summary.csv`, the climate-detection evaluation table, two SVG
figures, `findings_and_limitations.md`, and input hashes in
`results_metadata.json`. The evidence includes disclosure text and remains
local and Git-ignored.

User-reviewed labels and assistant-assessed candidates are separate; an AI
review is not an independent human annotation. Some flags need additional
context; model negatives were not comprehensively reviewed, and rule-generated
specificity and financial labels for AI candidates are unvalidated. Company
counts are not risk rankings or prevalence estimates. The convenience sample
is heavily software-focused: 17 of 20 industry labels contain "software".
Its composition and selection based on report availability limit generalization
to PE targets overall. A disclosure signal does not prove actual financial loss
or a risk hidden from investors or ESG ratings; financial-relevance labels are
qualitative, not measured impacts. The held-out cohorts
cannot support reliable overall precision, F1 or macro-F1. Historical ESG
scores were unavailable, so no ESG-score comparison is claimed. Verify cited
passages and surrounding context, document dates, and reuse rights before
publication.

## Additional PE climate-detection evaluation

`prepare-pe-review` prepares a blinded climate-relevance-only annotation sheet
from the four previously held-out companies, using the existing training split
and frozen model settings:

```powershell
python -m pe_esg_analysis.cli prepare-pe-review --budget 400
```

The command reproduces the TF-IDF baseline, checks its and ClimateBERT's
predictions against the previous comparison, excludes exact normalized text
seen in any training-company disclosure or previously annotated test/challenge
record, and samples without replacement. The two strata are passages flagged by
either model and passages flagged by neither. The design prioritizes reviewing
model flags while sampling both strata, recording the population and sample
sizes for weighting. The sheet is shuffled and does not display predictions;
the separate frozen-design file contains them and should not be consulted
while labeling. The output directory must not already exist.

In `review_sheet.csv`, leave the source columns unchanged. Mark
`is_climate_related` as `yes` or `no` and `annotation_status` as `complete`.
Use the climate-relevance rule in the codebook above, including the convention
for explicitly named weather hazards. If a passage cannot be judged, leave
its climate label blank, set the status to `unusable`, and explain in `notes`.
Do not use the assistant's previous reviews as labels. Additional independent
annotation and adjudication would strengthen the study.

After every record has a status, run:

```powershell
python -m pe_esg_analysis.cli evaluate-pe-review
```

The command checks the source text and frozen sampling design, refuses to
overwrite results, and reports inverse-probability-weighted climate-detection
accuracy, precision, recall and F1 for both models where defined. Unusable
records are excluded rather than labeled negative; the estimates concern
usable eligible records, not the full corpus. Report usable and positive
support, exclusions, sampling weights and limitations alongside the metrics.
The same four companies and some model-flagged texts were previously examined,
so this is not a fresh company-level test or fully independent annotation.
Few positives, correlated passages, and only four companies may make results
highly uncertain. No confidence intervals are calculated. Keep the models and
thresholds fixed during review, and do not claim that these metrics validate
the downstream categories or financial-relevance rules.

### Results of the 400-passage review

The review contained 400 completed labels: **1 climate-positive and 399
climate-negative**. The eligible population contained 4,570 extracted passages
from the four previously held-out companies. It excludes exact normalized text
found in any training-company disclosure or previously annotated test/challenge
record. All 58 passages flagged by either model were sampled, along with 342
randomly selected passages from the 4,512 flagged by neither model. The table
uses inverse sampling-probability weights of 1 and 4,512/342, respectively.

| Model | Accuracy | Positive precision | Positive recall | Positive F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClimateBERT | 98.86% | 1.89% | 100.00% | 3.70% | 51.57% |
| TF-IDF | 99.85% | 0.00% | 0.00% | 0.00% | 49.96% |

These are exploratory point estimates for climate-relevance detection on the
eligible passages, not estimates for all PE targets or the entire corpus.
Recall and F1 depend on a single reviewed positive, so they are highly
uncertain; accuracy is dominated by negatives. No confidence intervals were
calculated. The four companies and ClimateBERT-flagged passages had been
examined before this review, and independent annotation and adjudication are
still needed. The sole positive's extracted text is garbled; an AI assistant
visually checked the source PDF's physical page 151 and found the relevant
climate-related heading. This does not establish an actual financial loss.

The local [metrics CSV](../outputs/pe_evaluation_400/metrics.csv) and
[evaluation JSON](../outputs/pe_evaluation_400/evaluation.json) contain the
weighted confusion counts, unrounded metrics, input hashes and limitations.
Those outputs and the reviewed sheet are Git-ignored; this documentation does
not make the underlying data available in a fresh clone.
