from types import SimpleNamespace

import pytest

from pe_esg_analysis.climatebert import ClimateBertDetector, MODEL_ID, MODEL_REVISION
from pe_esg_analysis.classifiers import DisclosureClassifiers


def test_climatebert_uses_model_labels_and_rejects_long_inputs(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.ones((1, int(text)), dtype=torch.long)}

    class Model:
        config = SimpleNamespace(id2label={0: "yes", 1: "no"}, max_position_embeddings=514)

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.0]]))

    def tokenizer_loader(name, revision):
        calls.append((name, revision))
        return Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained",
        lambda name, revision: Model(),
    )
    detector = ClimateBertDetector()
    assert calls == [(MODEL_ID, MODEL_REVISION)]
    climate, confidence = DisclosureClassifiers(climate_detector=detector).detect_climate("4")
    assert climate is True
    assert confidence == pytest.approx(0.880797, rel=1e-5)
    detector.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))
    detector.model.config = SimpleNamespace(max_position_embeddings=514)
    climate, confidence = detector.predict_one("4")
    assert climate is False
    assert confidence == pytest.approx(0.880797, rel=1e-5)
    with pytest.raises(ValueError, match="limit is 512"):
        detector.predict_one("513")
