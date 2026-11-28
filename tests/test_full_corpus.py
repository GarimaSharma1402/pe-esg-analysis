import json

import pandas as pd
import pytest

from pe_esg_analysis import full_corpus


def test_full_corpus_outputs_provenance_and_rejects_overwrite(tmp_path, monkeypatch):
    class Detector:
        def predict_many(self, texts, batch_size):
            return [(text.startswith("Climate"), 0.9) for text in texts]

    monkeypatch.setattr(full_corpus, "ClimateBertDetector", Detector)
    companies = tmp_path / "companies.csv"
    disclosures = tmp_path / "disclosures.csv"
    output = tmp_path / "results"
    pd.DataFrame([
        {"company_id": name, "company_name": name, "acquisition_announcement_date": "2024-01-01"}
        for name in ("a", "b")
    ]).to_csv(companies, index=False)
    pd.DataFrame([
        {"record_id": f"{company}{i}", "company_id": company, "document_id": company,
         "document_date": "2023-01-01", "source": "example", "page": "1", "text": text}
        for company in ("a", "b") for i, text in enumerate(("Climate risk.", "Other disclosure."))
    ]).to_csv(disclosures, index=False)
    full_corpus.run_full_corpus(companies, disclosures, output, batch_size=2)
    predictions = pd.read_csv(output / "predictions.csv")
    profiles = pd.read_csv(output / "company_profiles.csv")
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert len(predictions) == 4 and len(profiles) == 2
    assert profiles["model_climate_positive_records"].tolist() == [1, 1]
    assert "total_relevance_score" not in profiles
    assert metadata["status"] == "complete" and metadata["unique_cleaned_texts"] == 2
    assert len(pd.read_csv(output / "by_company" / "a.csv")) == 2
    with pytest.raises(FileExistsError, match="overwrite"):
        full_corpus.run_full_corpus(companies, disclosures, output)


def test_full_corpus_rejects_post_announcement_records(tmp_path):
    companies = tmp_path / "companies.csv"
    disclosures = tmp_path / "disclosures.csv"
    pd.DataFrame([{"company_id": "a", "company_name": "A", "acquisition_announcement_date": "2024-01-01"}]).to_csv(companies, index=False)
    pd.DataFrame([{"record_id": "x", "company_id": "a", "document_id": "a",
                   "document_date": "2024-01-01", "source": "example", "page": "1",
                   "text": "Climate risk."}]).to_csv(disclosures, index=False)
    with pytest.raises(ValueError, match="on or after"):
        full_corpus.run_full_corpus(companies, disclosures, tmp_path / "results")
