import json

import pandas as pd
import pytest

from pe_esg_analysis.annotation import prepare_annotation_set, reviewed_annotations


def test_company_split_excludes_shared_text_and_preserves_blank_labels(tmp_path):
    companies = pd.DataFrame([
        {"company_id": name, "company_name": name, "acquisition_announcement_date": "2024-01-01"}
        for name in ("a", "b", "c")
    ])
    rows = []
    for company in companies["company_id"]:
        for index, text in enumerate(("Shared boilerplate.", f"{company} text one.", f"{company} text two.")):
            rows.append({
                "record_id": f"{company}{index}", "company_id": company,
                "document_id": company, "document_date": "2023-01-01",
                "source": "https://example.test", "page": "1", "text": text,
            })
    source, company_file = tmp_path / "disclosures.csv", tmp_path / "companies.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    companies.to_csv(company_file, index=False)
    out = tmp_path / "annotations"
    metadata = prepare_annotation_set(source, company_file, out, per_company=2, test_companies=1)
    train = pd.read_csv(out / "train_annotations.csv", keep_default_na=False)
    test = pd.read_csv(out / "test_annotations.csv", keep_default_na=False)
    assert len(train) == 4 and len(test) == 2
    assert set(train.company_id).isdisjoint(test.company_id)
    assert set(train.text).isdisjoint(test.text)
    assert train["is_climate_related"].eq("").all()
    assert train["annotation_status"].eq("").all()
    assert metadata["cross_split_duplicate_records_excluded"] == 3
    assert json.loads((out / "sampling_metadata.json").read_text())["seed"] == 42
    replay = tmp_path / "replay"
    prepare_annotation_set(source, company_file, replay, per_company=2, test_companies=1)
    assert pd.read_csv(replay / "test_annotations.csv")["record_id"].tolist() == test["record_id"].tolist()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_annotation_set(source, company_file, out, per_company=2, test_companies=1)

    companies.loc[0, "acquisition_announcement_date"] = "2023-01-01"
    companies.to_csv(company_file, index=False)
    with pytest.raises(ValueError, match="cutoff"):
        prepare_annotation_set(source, company_file, tmp_path / "other", per_company=2, test_companies=1)


def test_reviewed_annotations_requires_complete_conditional_labels():
    rows = pd.DataFrame([
        {"record_id": "a", "annotation_status": "complete", "is_climate_related": "no",
         "is_specific": "", "risk_category": "", "financial_relevance": "", "notes": ""},
        {"record_id": "b", "annotation_status": "unusable", "is_climate_related": "",
         "is_specific": "", "risk_category": "", "financial_relevance": "", "notes": "fragment"},
    ])
    assert reviewed_annotations(rows)["record_id"].tolist() == ["a"]
    rows.loc[0, "annotation_status"] = ""
    with pytest.raises(ValueError, match="status"):
        reviewed_annotations(rows)
    rows.loc[0, "annotation_status"] = "complete"
    rows.loc[0, "is_climate_related"] = "yes"
    with pytest.raises(ValueError, match="Invalid binary label"):
        reviewed_annotations(rows)
    rows.loc[0, ["is_specific", "risk_category", "financial_relevance"]] = ["yes", "physical", "high"]
    assert reviewed_annotations(rows)["record_id"].tolist() == ["a"]
    with pytest.raises(ValueError, match="unique"):
        reviewed_annotations(pd.concat([rows, rows.iloc[:1]]))
