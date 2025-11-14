from __future__ import annotations

from dataclasses import dataclass, InitVar, field

import torch
from torch import Tensor, nn

from methods.base import BaseTTA


@dataclass
class VarianceMinimizationEM(BaseTTA):
    """Entropy-minimisation-style VBLL adaptation that suppresses predictive variance."""

    vbll_head: nn.Module | None = field(default=None)
    variance_weight: float = 1.0


    def __post_init__(self, compile_model: dict | None):
        super().__post_init__(compile_model)
        if self.vbll_head is None:
            raise ValueError("vbll_head must be provided for VarianceMinimizationEM")
        self.vbll_head = self.vbll_head.to(self.device)
        for param in self.vbll_head.parameters():
            param.requires_grad_(False)
        self.vbll_head.eval()

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.vbll_head is None:
            raise ValueError("vbll_head must be provided for VarianceMinimizationEM")
        
        feature_extractor = getattr(self.net, "feature")
        predictor = getattr(self.net, "predict_from_feature")

        features = feature_extractor(x)
        vbll_out = self.vbll_head(features)

        preds = predictor(features).view(-1)
        variance = vbll_out.predictive.variance.view(-1)

        loss = self.variance_weight * variance.mean()

        return {"y_pred": preds, "vbll_var": variance}, loss
