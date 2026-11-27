import csv
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")
transformers = pytest.importorskip("transformers")

from pe_esg_analysis.extraction import extract_corpus, split_for_model


class WordTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_split_preserves_all_words_and_limit():
    text = " ".join(f"word{i}" for i in range(25))
    parts = split_for_model(text, WordTokenizer(), limit=8)
    assert " ".join(parts) == text
    assert all(len(WordTokenizer()(part)["input_ids"]) <= 8 for part in parts)


def test_quality_check_rejects_garbled_text_but_accepts_tables():
    from pe_esg_analysis.extraction import readable

    assert not readable("garbled\x01text\x01" * 20)
    assert readable("Revenue 2020 2019 2018 100 200 300 400 500")


def test_extract_respects_ranges_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: WordTokenizer())
    pdf = pymupdf.open()
    for text in ["Proxy statement excluded.", "A facility faces significant flood risks.", "Later material excluded."]:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(tmp_path / "report.pdf")
    pdf.close()
    manifest = tmp_path / "manifest.csv"
    ranges = tmp_path / "ranges.csv"
    write_rows(manifest, [{
        "company_id": "example", "local_file": "report.pdf", "report_year": "2022",
        "filing_date": "2023-02-01", "acquisition_announcement_date": "2023-03-01",
        "report_url": "https://example.test/report.pdf",
    }])
    write_rows(ranges, [{"company_id": "example", "start_page": "2", "end_page": "2", "notes": "10-K"}])
    output, report = tmp_path / "output.csv", tmp_path / "report.json"
    extract_corpus(manifest, ranges, tmp_path, output, report)
    with output.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert len(records) == 1
    assert records[0]["text"] == "A facility faces significant flood risks."
    assert records[0]["record_id"] == "example_2022_p0002_0001"
    assert records[0]["page"] == "2"
    assert records[0]["document_date"] == "2023-02-01"
    assert records[0]["source"] == "https://example.test/report.pdf"
    assert json.loads(report.read_text())[0]["pages"][0]["status"] == "ok"

    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    rows[0]["acquisition_announcement_date"] = "2023-02-01"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="on or after"):
        extract_corpus(manifest, ranges, tmp_path, output, report)


def test_unreadable_pages_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: WordTokenizer())
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(tmp_path / "blank.pdf")
    pdf.close()
    write_rows(tmp_path / "manifest.csv", [{
        "company_id": "example", "local_file": "blank.pdf", "report_year": "2022",
        "filing_date": "2023-01-01", "acquisition_announcement_date": "2023-03-01",
        "report_url": "https://example.test/blank.pdf",
    }])
    write_rows(tmp_path / "ranges.csv", [{
        "company_id": "example", "start_page": "1", "end_page": "1", "notes": "10-K",
    }])
    with pytest.raises(RuntimeError, match="incomplete"):
        extract_corpus(tmp_path / "manifest.csv", tmp_path / "ranges.csv", tmp_path,
                       tmp_path / "output.csv", tmp_path / "report.json")
    result = json.loads((tmp_path / "report.json").read_text())
    assert result[0]["pages"][0]["status"] == "unreadable or blank"
