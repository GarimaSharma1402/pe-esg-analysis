"""Local inference with the published ClimateBERT paragraph detector."""

from __future__ import annotations

MODEL_ID = "climatebert/distilroberta-base-climate-detector"
MODEL_REVISION = "2c3bc660d45a59e31b35f5d3e365ee4f59fdf76c"


class ClimateBertDetector:
    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ClimateBERT requires the optional dependencies: "
                'pip install -e ".[transformers]"'
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model.eval()
        labels = {str(label).lower(): int(index) for index, label in self.model.config.id2label.items()}
        if set(labels) != {"no", "yes"}:
            raise ValueError(f"Unexpected ClimateBERT labels: {labels}")
        self.yes_index = labels["yes"]

    def predict_one(self, text: str) -> tuple[bool, float]:
        if not text.strip():
            raise ValueError("ClimateBERT requires a non-empty paragraph")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        # RoBERTa's position embeddings include two reserved positions.
        max_tokens = self.model.config.max_position_embeddings - 2
        token_count = encoded["input_ids"].shape[1]
        if token_count > max_tokens:
            raise ValueError(
                f"ClimateBERT input has {token_count} tokens; limit is {max_tokens}. "
                "Split the disclosure into shorter paragraphs before analysis."
            )
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)[0]
        positive_probability = float(probabilities[self.yes_index].item())
        is_climate = positive_probability >= 0.5
        return is_climate, positive_probability if is_climate else 1.0 - positive_probability
