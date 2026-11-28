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

    def predict_many(self, texts: list[str], batch_size: int = 16) -> list[tuple[bool, float]]:
        """Predict without truncation, grouping similarly sized texts to reduce padding."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        unique = list(dict.fromkeys(texts))
        if any(not text.strip() for text in unique):
            raise ValueError("ClimateBERT requires non-empty paragraphs")
        max_tokens = self.model.config.max_position_embeddings - 2
        tokenized = self.tokenizer(unique, truncation=False, add_special_tokens=True) if unique else {"input_ids": []}
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        if any(length > max_tokens for length in lengths):
            raise ValueError(f"ClimateBERT input exceeds the {max_tokens}-token limit")
        order = sorted(range(len(unique)), key=lengths.__getitem__)
        predictions: dict[str, tuple[bool, float]] = {}
        for start in range(0, len(order), batch_size):
            group = [unique[index] for index in order[start:start + batch_size]]
            encoded = self.tokenizer(group, return_tensors="pt", padding=True, truncation=False)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)[:, self.yes_index].tolist()
            for text, probability in zip(group, probabilities, strict=True):
                # Resolve potential floating-point differences at the classification threshold.
                if abs(probability - 0.5) < 0.00001:
                    predictions[text] = self.predict_one(text)
                else:
                    climate = probability >= 0.5
                    predictions[text] = climate, probability if climate else 1.0 - probability
        return [predictions[text] for text in texts]
