"""OSL predictor model architectures.

L2: Lightweight transformer — uses a small pre-trained model (TinyBERT or
    DistilBERT with truncation to 128 tokens) with a prediction head.
    Designed for low-overhead prediction at request routing time.

L3: Full DistilBERT — processes the full prompt (up to 512 tokens) for
    maximum accuracy.

Both support two prediction modes:
  - Classification: predict OSL bucket (discrete bins)
  - Quantile regression: predict multiple quantiles of P(OSL | prompt, model)
    for risk-aware scheduling (e.g., use p90 for SLA-sensitive routing)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class QuantileHead(nn.Module):
    """Predicts multiple quantiles of the OSL distribution.

    Outputs one value per quantile. Trained with pinball (quantile) loss.
    """

    def __init__(self, hidden_size: int, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(quantiles)),
        )

    def forward(self, h):
        return self.head(h)  # (batch, n_quantiles)


class ClassificationHead(nn.Module):
    """Predicts OSL bucket as a classification task."""

    def __init__(self, hidden_size: int, n_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, h):
        return self.head(h)  # (batch, n_classes)


class OSLPredictor(nn.Module):
    """Output Sequence Length predictor.

    Architecture:
        [Tokenizer] -> [Transformer encoder] -> [CLS pooling]
            -> [model_embedding (optional)] -> [prediction head]

    Args:
        backbone: HuggingFace model name (e.g. "distilbert-base-uncased").
        max_length: Max token length for the backbone.
        n_classes: Number of OSL buckets (for classification mode).
        quantiles: Tuple of quantile levels (for quantile mode).
        mode: "classification", "quantile", or "both".
        n_models: Number of distinct LLM models (for model-conditional
                  prediction). If > 0, a learned embedding is concatenated
                  to the CLS vector.
        n_datasets: Number of distinct datasets/task types. If > 0, a
                    learned embedding is concatenated to the CLS vector.
        freeze_backbone_layers: Number of bottom transformer layers to
                                freeze (0 = fine-tune all).
    """

    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        max_length: int = 512,
        n_classes: int = 8,
        quantiles: tuple = (0.1, 0.25, 0.5, 0.75, 0.9),
        mode: str = "both",
        n_models: int = 0,
        n_datasets: int = 0,
        freeze_backbone_layers: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        self.quantiles = quantiles

        # Backbone
        config = AutoConfig.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone, config=config)
        hidden_size = config.hidden_size

        # Optional: freeze bottom layers
        if freeze_backbone_layers > 0:
            self._freeze_layers(freeze_backbone_layers)

        # Model-conditional embedding
        self.n_models = n_models
        embed_extra = 0
        if n_models > 0:
            self.model_embed = nn.Embedding(n_models, 32)
            embed_extra += 32
        else:
            self.model_embed = None

        # Dataset/task-type embedding
        self.n_datasets = n_datasets
        if n_datasets > 0:
            self.dataset_embed = nn.Embedding(n_datasets, 32)
            embed_extra += 32
        else:
            self.dataset_embed = None

        head_input_size = hidden_size + embed_extra

        # Prediction heads
        if mode in ("classification", "both"):
            self.cls_head = ClassificationHead(head_input_size, n_classes)
        else:
            self.cls_head = None

        if mode in ("quantile", "both"):
            self.q_head = QuantileHead(head_input_size, quantiles)
        else:
            self.q_head = None

    def _freeze_layers(self, n_layers: int):
        """Freeze the embedding and first n_layers transformer layers."""
        # Freeze embeddings
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        # Freeze transformer layers
        if hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        else:
            return
        for i, layer in enumerate(layers):
            if i < n_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, model_ids=None, dataset_ids=None):
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            model_ids: (batch,) integer model IDs (optional).
            dataset_ids: (batch,) integer dataset IDs (optional).

        Returns:
            dict with keys "cls_logits" and/or "quantiles" depending on mode.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # CLS pooling: use first token
        h = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        # Concatenate model embedding if available
        if self.model_embed is not None and model_ids is not None:
            m_emb = self.model_embed(model_ids)  # (batch, 32)
            h = torch.cat([h, m_emb], dim=-1)

        # Concatenate dataset embedding if available
        if self.dataset_embed is not None and dataset_ids is not None:
            d_emb = self.dataset_embed(dataset_ids)  # (batch, 32)
            h = torch.cat([h, d_emb], dim=-1)

        result = {}
        if self.cls_head is not None:
            result["cls_logits"] = self.cls_head(h)
        if self.q_head is not None:
            result["quantiles"] = self.q_head(h)

        return result


def build_l2_predictor(n_classes=8, n_models=0, n_datasets=0, **kwargs):
    """L2: Lightweight predictor — DistilBERT truncated to 128 tokens,
    bottom 4 layers frozen. Fast inference (~5ms)."""
    return OSLPredictor(
        backbone="distilbert-base-uncased",
        max_length=128,
        n_classes=n_classes,
        mode="both",
        n_models=n_models,
        n_datasets=n_datasets,
        freeze_backbone_layers=4,
        **kwargs,
    )


def build_l3_predictor(n_classes=8, n_models=0, n_datasets=0, **kwargs):
    """L3: Full DistilBERT — processes up to 512 tokens, all layers
    fine-tuned. Higher accuracy, ~20ms inference."""
    return OSLPredictor(
        backbone="distilbert-base-uncased",
        max_length=512,
        n_classes=n_classes,
        mode="both",
        n_models=n_models,
        n_datasets=n_datasets,
        freeze_backbone_layers=0,
        **kwargs,
    )
