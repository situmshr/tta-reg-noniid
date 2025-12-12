from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from methods.base import BaseTTA


@dataclass
class AdaptiveBatchNorm(BaseTTA):
    """AdaBN: run BatchNorm layers with mini-batch statistics during inference."""

    def __post_init__(self, compile_model: dict | None, val_dataset=None, target_names=None):
        super().__post_init__(compile_model, val_dataset, target_names)
        self.train_mode = False  # keep non-BN layers in eval mode
        self._bn_layers: Tuple[_BatchNorm, ...] = tuple(self._collect_bn_layers(self.net))
        if not self._bn_layers:
            raise ValueError("AdaptiveBatchNorm requires BatchNorm layers in the network.")
        for layer in self._bn_layers:
            for param in layer.parameters():
                param.requires_grad_(False)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor | None]:
        self.net.eval()
        for layer in self._bn_layers:
            layer.train()  # force batch statistics

        with torch.no_grad():
            y_pred = self.net(x).flatten()

        return {"y_pred": y_pred}, None

    @staticmethod
    def _collect_bn_layers(module: nn.Module) -> Iterable[_BatchNorm]:
        for layer in module.modules():
            if isinstance(layer, _BatchNorm):
                yield layer
