from dataclasses import dataclass, InitVar, field
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss


def calc_weighted_stats(features: Tensor, weights: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    """Compute weighted mean/variance for per-dimension PCA scores."""
    sum_weights = weights.sum() + eps
    weighted_mean = (features * weights).sum(dim=0) / sum_weights
    variance_term = (features - weighted_mean).pow(2)
    weighted_var = (variance_term * weights).sum(dim=0) / sum_weights
    return weighted_mean, weighted_var, sum_weights


class TensorCallable(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class KLCallable(Protocol):
    def __call__(self, m1: Tensor, v1: Tensor, m2: Tensor, v2: Tensor) -> Tensor: ...


@dataclass
class WeightedSignificantSubspaceAlignment(BaseTTA):
    """Weighted SSA that re-weights samples based on distance in PCA space."""

    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None
    weight_bias: float = 1e-6
    weight_exp: float = 1.0
    temperature: float = 1.0
    feature_extractor: TensorCallable | None = field(init=False, default=None, repr=False)
    predictor: TensorCallable | None = field(init=False, default=None, repr=False)
    loss_fn: KLCallable | None = field(init=False, default=None, repr=False)
    dim_weight: Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self,
                      compile_model: dict | None,
                      val_dataset,
                      target_names,
                      pc_config: dict | None,
                      loss_config: dict | None):
        self._pc_config = dict(pc_config or {})
        self._loss_config = dict(loss_config or {})
        super().__post_init__(compile_model, val_dataset, target_names)
        self._init_subspace()

    def _init_subspace(self) -> None:
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.mean = mean.to(self.device)
        self.basis = basis.to(self.device)
        self.var = var.to(self.device)

        self.loss_fn = lambda m1, v1, m2, v2: diagonal_gaussian_kl_loss(
            m1, v1, m2, v2, dim_reduction="none", **self._loss_config)

        with torch.no_grad():
            regressor_weight = self._get_regressor_weight()
            dim_weight = torch.abs(regressor_weight @ self.basis).flatten() + self.weight_bias
            self.dim_weight = dim_weight.pow(self.weight_exp)
            print(f"WSSA dim_weight: {self.dim_weight}")

        feature = getattr(self.net, "feature", None)
        predictor = getattr(self.net, "predict_from_feature", None)
        if not callable(feature) or not callable(predictor):
            raise AttributeError("Net must define callable feature/predict_from_feature methods.")
        self.feature_extractor = cast(TensorCallable, feature)
        self.predictor = cast(TensorCallable, predictor)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.feature_extractor is None or self.predictor is None:
            raise RuntimeError("WSSA feature extractor is not initialized.")
        feature = self.feature_extractor(x)
        y_pred = self.predictor(feature)

        f_pc = (feature - self.mean) @ self.basis  # (B, d)
        dist_sq = (f_pc.pow(2) / self.var).sum(dim=1, keepdim=True)
        sample_weights = torch.exp(-0.5 * dist_sq / max(self.temperature, 1e-6))
        sample_weights = torch.clamp(sample_weights, min=1e-8)

        f_pc_mean, f_pc_var, sum_weights = calc_weighted_stats(f_pc, sample_weights)

        zeros = torch.zeros_like(f_pc_mean)
        if self.loss_fn is None or self.dim_weight is None:
            raise RuntimeError("WSSA loss function is not initialized.")
        kl_loss = self.loss_fn(f_pc_mean, f_pc_var, zeros, self.var) \
            + self.loss_fn(zeros, self.var, f_pc_mean, f_pc_var)

        kl_loss = (kl_loss * self.dim_weight).sum()
        batch_reliability = sum_weights / x.size(0)
        final_loss = kl_loss * batch_reliability

        return {
            "y_pred": y_pred,
            "feat_pc": f_pc,
            "reliability": batch_reliability,
        }, final_loss

    def _get_regressor_weight(self) -> Tensor:
        regressor_module = getattr(self.net, "regressor", None)
        if regressor_module is None or not isinstance(regressor_module, nn.Module):
            raise AttributeError("Net must expose a regressor module with weight parameter.")
        weight = getattr(regressor_module, "weight", None)
        if not isinstance(weight, Tensor):
            raise AttributeError("Regressor module is missing weight tensor.")
        return weight
