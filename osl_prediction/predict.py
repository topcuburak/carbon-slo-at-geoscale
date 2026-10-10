"""Inference-time OSL predictor for use in the scheduling pipeline.

Loads a trained checkpoint and predicts OSL for incoming prompts.
Designed to be called at request routing time with minimal overhead.

Usage:
    from osl_prediction.predict import OSLPredictorService

    svc = OSLPredictorService("osl_prediction/checkpoints/L1_cls/best_model.pt")
    result = svc.predict("Summarize the following report...", model="llama3_8B")
    print(result)
    # {'bucket': 5, 'quantiles': {0.1: 120, 0.25: 200, 0.5: 350, 0.75: 500, 0.9: 700}}
"""

import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from osl_prediction.models import build_l2_predictor, build_l3_predictor


class OSLPredictorService:
    """Stateful OSL predictor for serving-time use.

    Loads model weights and tokenizer once, provides fast .predict() calls.
    Thread-safe for concurrent requests (model is in eval mode, no state mutation).
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Load a trained OSL predictor from checkpoint.

        Args:
            checkpoint_path: Path to best_model.pt saved by train.py.
            device: "cuda", "cpu", or "auto".
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        config = ckpt["config"]
        self.model_to_id = ckpt["model_to_id"]
        self.bin_edges = np.array(ckpt["bin_edges"])
        self.quantile_levels = ckpt.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])

        level = config["level"]
        n_classes = len(self.bin_edges) - 1
        n_models = len(self.model_to_id)
        quantiles = tuple(self.quantile_levels)

        if level == "L2":
            self.model = build_l2_predictor(
                n_classes=n_classes, n_models=n_models, quantiles=quantiles,
            )
            self.max_length = 128
        else:
            self.model = build_l3_predictor(
                n_classes=n_classes, n_models=n_models, quantiles=quantiles,
            )
            self.max_length = 512

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @torch.no_grad()
    def predict(self, prompt: str, model: Optional[str] = None) -> dict:
        """Predict OSL for a single prompt.

        Args:
            prompt: The input prompt text.
            model: LLM model key (e.g. "llama3_8B"). If None, uses first model.

        Returns:
            dict with:
                bucket: predicted OSL bucket index
                bucket_range: (low, high) token range for the predicted bucket
                quantiles: dict mapping quantile level -> predicted token count
                predict_ms: inference time in milliseconds
        """
        t0 = time.monotonic()

        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        model_id = 0
        if model and model in self.model_to_id:
            model_id = self.model_to_id[model]
        model_ids = torch.tensor([model_id], dtype=torch.long, device=self.device)

        out = self.model(input_ids, attention_mask, model_ids)

        result = {}

        if "cls_logits" in out:
            bucket = out["cls_logits"].argmax(dim=1).item()
            result["bucket"] = bucket
            result["bucket_range"] = (
                float(self.bin_edges[bucket]),
                float(self.bin_edges[min(bucket + 1, len(self.bin_edges) - 1)]),
            )

        if "quantiles" in out:
            q_vals = out["quantiles"].squeeze(0).cpu().numpy()
            result["quantiles"] = {
                float(q): max(0, float(v))
                for q, v in zip(self.quantile_levels, q_vals)
            }

        result["predict_ms"] = (time.monotonic() - t0) * 1000
        return result

    @torch.no_grad()
    def predict_batch(self, prompts: list, models: Optional[list] = None) -> list:
        """Predict OSL for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            models: List of model keys (same length as prompts), or None.

        Returns:
            List of prediction dicts (same format as predict()).
        """
        t0 = time.monotonic()

        enc = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if models:
            model_ids = torch.tensor(
                [self.model_to_id.get(m, 0) for m in models],
                dtype=torch.long, device=self.device,
            )
        else:
            model_ids = torch.zeros(len(prompts), dtype=torch.long,
                                    device=self.device)

        out = self.model(input_ids, attention_mask, model_ids)
        elapsed_ms = (time.monotonic() - t0) * 1000
        per_request_ms = elapsed_ms / len(prompts)

        results = []
        for i in range(len(prompts)):
            r = {}
            if "cls_logits" in out:
                bucket = out["cls_logits"][i].argmax().item()
                r["bucket"] = bucket
                r["bucket_range"] = (
                    float(self.bin_edges[bucket]),
                    float(self.bin_edges[min(bucket + 1, len(self.bin_edges) - 1)]),
                )
            if "quantiles" in out:
                q_vals = out["quantiles"][i].cpu().numpy()
                r["quantiles"] = {
                    float(q): max(0, float(v))
                    for q, v in zip(self.quantile_levels, q_vals)
                }
            r["predict_ms"] = per_request_ms
            results.append(r)

        return results
