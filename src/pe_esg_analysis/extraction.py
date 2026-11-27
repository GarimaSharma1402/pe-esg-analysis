"""Extract auditable page-level disclosure records from selected PDF ranges."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from .climatebert import MODEL_ID, MODEL_REVISION
from .preprocessing import clean_text


FIELDS = ("record_id", "company_id", "document_id", "document_date", "source", "page", "text")


def readable(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    controls = sum(not c.isprintable() or c == "\ufffd" for c in chars)
    return controls / len(chars) < 0.02 and sum(c.isalpha() for c in chars) / len(chars) > 0.1


def split_for_model(text: str, tokenizer, limit: int = 512) -> list[str]:
    """Split at word boundaries without truncating or dropping any words."""
    words = text.split()
    chunks = []
    while words:
        if len(tokenizer(" ".join(words), add_special_tokens=True)["input_ids"]) <= limit:
            chunks.append(" ".join(words))
            break
        low, high = 1, len(words)
        while low < high:
            mid = (low + high + 1) // 2
            if len(tokenizer(" ".join(words[:mid]), add_special_tokens=True)["input_ids"]) <= limit:
                low = mid
            else:
                high = mid - 1
        if len(tokenizer(" ".join(words[:low]), add_special_tokens=True)["input_ids"]) > limit:
            raise ValueError("A single word exceeds the ClimateBERT token limit")
        chunks.append(" ".join(words[:low]))
        words = words[low:]
    return chunks


def extract_corpus(manifest_path, ranges_path, pdf_dir, output_path, report_path) -> None:
    try:
        import pymupdf
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install PDF and tokenizer dependencies: pip install -e ".[pdf,transformers]"') from exc

    with open(manifest_path, encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    with open(ranges_path, encoding="utf-8-sig", newline="") as file:
        ranges = list(csv.DictReader(file))
    by_company = {row["company_id"]: row for row in ranges}
    if (len(by_company) != len(ranges)
            or len(manifest) != len({row["company_id"] for row in manifest})
            or set(by_company) != {row["company_id"] for row in manifest}):
        raise ValueError("Extraction ranges must contain exactly one entry per manifest company")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    pymupdf.TOOLS.mupdf_display_errors(False)
    records, report = [], []
    for row in manifest:
        company = row["company_id"]
        selected = by_company[company]
        result = {"company_id": company, "file": row["local_file"], "pages": [], "notes": selected["notes"]}
        incomplete_reason = selected.get("incomplete_reason") or ""
        if incomplete_reason:
            result["incomplete_reason"] = incomplete_reason
        report.append(result)
        if not selected["start_page"] or not selected["end_page"]:
            result["status"] = "excluded: no verified page range"
            continue
        if date.fromisoformat(row["filing_date"]) >= date.fromisoformat(row["acquisition_announcement_date"]):
            raise ValueError(f"{company}: filing is on or after acquisition announcement")
        path = Path(pdf_dir) / row["local_file"]
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with pymupdf.open(path) as document:
            start, end = int(selected["start_page"]), int(selected["end_page"])
            if not 1 <= start <= end <= len(document):
                raise ValueError(f"{company}: invalid physical PDF page range {start}-{end}")
            result["status"] = "needs review" if incomplete_reason else "extracted"
            for number in range(start, end + 1):
                page = document[number - 1]
                blocks = [clean_text(block[4]) for block in page.get_text("blocks", sort=True) if block[4].strip()]
                text = " ".join(blocks)
                if not readable(text):
                    result["pages"].append({"page": number, "status": "unreadable or blank", "method": "pdf"})
                    result["status"] = "needs review"
                    continue
                count = 0
                for block in blocks:
                    if not block:
                        continue
                    for part in split_for_model(block, tokenizer):
                        count += 1
                        records.append({
                            "record_id": f"{company}_{row['report_year']}_p{number:04d}_{count:04d}",
                            "company_id": company, "document_id": Path(row["local_file"]).stem,
                            "document_date": row["filing_date"], "source": row["report_url"],
                            "page": number, "text": part,
                        })
                result["pages"].append({"page": number, "status": "ok", "method": "pdf", "records": count})
        print(f"{company}: {result['status']}", flush=True)
    output = Path(output_path)
    report_file = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(item["status"] != "extracted" for item in report):
        raise RuntimeError(f"Extraction incomplete; see {report_file}")
