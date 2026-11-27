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
