"""Loss functions for OSL prediction.

Pinball (quantile) loss allows the model to predict the full conditional
distribution of OSL, enabling risk-aware scheduling decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PinballLoss(nn.Module):
    """Quantile (pinball) loss for distribution prediction.

    For quantile tau, the loss is:
        L(y, y_hat) = tau * max(y - y_hat, 0) + (1 - tau) * max(y_hat - y, 0)

    Averaged across all quantiles and batch elements.
    """

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, n_quantiles) predicted quantile values.
            target: (batch,) true OSL values.
        """
        target = target.unsqueeze(1).expand_as(pred)  # (batch, n_quantiles)
        errors = target - pred  # (batch, n_quantiles)
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        )
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined classification + quantile regression loss.

    total_loss = cls_weight * CE_loss + q_weight * pinball_loss

    The classification loss encourages coarse bucket accuracy while
    the quantile loss learns the fine-grained distribution.
    """

    def __init__(
        self,
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        cls_weight: float = 1.0,
        q_weight: float = 1.0,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.q_weight = q_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.pinball_loss = PinballLoss(quantiles)

    def forward(self, cls_logits, q_pred, cls_target, q_target):
        """
        Args:
            cls_logits: (batch, n_classes) classification logits.
            q_pred: (batch, n_quantiles) quantile predictions.
            cls_target: (batch,) integer bucket labels.
            q_target: (batch,) float OSL values (for quantile loss).
        """
        loss = 0.0
        components = {}

        if cls_logits is not None:
            ce = self.ce_loss(cls_logits, cls_target)
            loss = loss + self.cls_weight * ce
            components["ce"] = ce.item()

        if q_pred is not None:
            pb = self.pinball_loss(q_pred, q_target)
            loss = loss + self.q_weight * pb
            components["pinball"] = pb.item()

        components["total"] = loss.item()
        return loss, components
