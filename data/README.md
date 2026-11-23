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

Some PDFs also contain proxy statements or other material. Before analysis,
identify the 10-K pages and separately verify the publication dates of any
additional material; do not assign the 10-K filing date to the whole PDF
without checking. The Model N and PowerSchool PDFs open and display correctly,
but their text extracted as garbled characters with two PDF libraries. They
need a different source or OCR and quality checks before NLP analysis.

For Anaplan and Medallia, the archive's URL label says 2020, but the PDFs
contain fiscal-year 2021 10-Ks; the filenames and manifest reflect their
actual report periods. This corpus is not yet a labeled dataset, and it has
not been converted into the paragraph-level `disclosures.csv` required by
the pipeline.
