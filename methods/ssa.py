from __future__ import annotations

from dataclasses import dataclass, InitVar

import torch
from torch import Tensor

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss

@dataclass
class SignificantSubspaceAlignment(BaseTTA):
    """SSA: subspace alignment style TTA using PCA statistics."""

    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None
    weight_bias: float = 1e-6
    weight_exp: float = 1.0

    def __post_init__(self,
                      pc_config: dict | None,
                      loss_config: dict | None,
                      compile_model: dict | None):
        self._pc_config = dict(pc_config or {})
        self._loss_config = dict(loss_config or {})
        self.feature_extractor = None
        super().__post_init__(compile_model)
        self._init_subspace()

    def _init_subspace(self) -> None:
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.mean = mean.to(self.device)
        self.basis = basis.to(self.device)
        self.var = var.to(self.device)

        self.loss_fn = lambda m1, v1, m2, v2: diagonal_gaussian_kl_loss(
            m1, v1, m2, v2, dim_reduction="none", **self._loss_config)

        with torch.no_grad():
            regressor_weight = self.net.regressor.weight if isinstance(self.net.regressor.weight, Tensor) else self.net.regressor.weight.data
            dim_weight = torch.abs(
                regressor_weight @ self.basis).flatten() + self.weight_bias
            self.dim_weight = dim_weight.pow(self.weight_exp)
            print(f"SSA dim_weight: {self.dim_weight}")

        self.feature_extractor = getattr(self.net, "feature")

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        f_pc = (feature - self.mean) @ self.basis  # (B, d)
        f_pc_mean = f_pc.mean(dim=0)
        f_pc_var = f_pc.var(dim=0, unbiased=False)

        zeros = torch.zeros_like(f_pc_mean)
        kl_loss = self.loss_fn(f_pc_mean, f_pc_var, zeros, self.var) \
            + self.loss_fn(zeros, self.var, f_pc_mean, f_pc_var)

        kl_loss = (kl_loss * self.dim_weight).sum()

        return {"y_pred": y_pred, "feat_pc": f_pc}, kl_loss
