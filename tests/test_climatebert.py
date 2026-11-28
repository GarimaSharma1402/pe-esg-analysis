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


def test_batched_predictions_preserve_order_deduplicate_and_reject_long_inputs():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, texts, return_tensors=None, **kwargs):
            single = isinstance(texts, str)
            lengths = [int(value) for value in ([texts] if single else texts)]
            if return_tensors:
                width = max(lengths)
                return {"input_ids": torch.tensor([
                    [1] * length + [0] * (width - length) for length in lengths
                ])}
            return {"input_ids": [[1] * length for length in lengths]}

    class Model:
        config = SimpleNamespace(max_position_embeddings=514)

        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids):
            self.calls += 1
            scores = input_ids.sum(dim=1).float() - 3.5
            return SimpleNamespace(logits=torch.stack((-scores, scores), dim=1))

    detector = ClimateBertDetector.__new__(ClimateBertDetector)
    detector.torch = torch
    detector.tokenizer = Tokenizer()
    detector.model = Model()
    detector.yes_index = 1
    texts = ["4", "2", "4", "6"]
    actual = detector.predict_many(texts, batch_size=2)
    assert detector.model.calls == 2
    expected = [detector.predict_one(text) for text in texts]
    assert [label for label, _ in actual] == [label for label, _ in expected]
    assert [confidence for _, confidence in actual] == pytest.approx([confidence for _, confidence in expected])
    with pytest.raises(ValueError, match="512-token limit"):
        detector.predict_many(["513"])
