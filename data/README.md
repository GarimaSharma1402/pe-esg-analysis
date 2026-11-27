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
