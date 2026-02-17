from dataclasses import dataclass, InitVar, field
from typing import Protocol, cast

import torch
from torch import Tensor
from torch import nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss

class TensorCallable(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class KLCallable(Protocol):
    def __call__(self, m1: Tensor, v1: Tensor, m2: Tensor, v2: Tensor) -> Tensor: ...


def _select_center_frame(x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    if x.dim() == 5:
        seq_len = x.size(1)
        center = seq_len // 2
        x = x[:, center]
        if y is not None and y.dim() == 3 and y.size(1) == seq_len:
            y = y[:, center]
    return x, y


@dataclass
class SSA(BaseTTA):
    """SSA: subspace alignment style TTA using PCA statistics."""

    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None
    weight_bias: float = 1.0
    weight_exp: float = 1.0
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
            dim_weight = torch.abs(regressor_weight @ self.basis).sum(dim=0)
            self.dim_weight = (dim_weight + self.weight_bias).pow(self.weight_exp)

        feature = getattr(self.net, "feature", None)
        predictor = getattr(self.net, "predict_from_feature", None)
        if not callable(feature) or not callable(predictor):
            raise AttributeError("Net must define callable feature/predict_from_feature methods.")
        self.feature_extractor = cast(TensorCallable, feature)
        self.predictor = cast(TensorCallable, predictor)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.feature_extractor is None or self.predictor is None:
            raise RuntimeError("SSA feature extractor is not initialized.")

        x, _ = _select_center_frame(x)

        feature = self.feature_extractor(x)
        y_pred = self.predictor(feature)

        f_pc = (feature - self.mean) @ self.basis  # (B, d)
        f_pc_mean = f_pc.mean(dim=0)
        f_pc_var = f_pc.var(dim=0, unbiased=False)

        zeros = torch.zeros_like(f_pc_mean)
        if self.loss_fn is None or self.dim_weight is None:
            raise RuntimeError("SSA loss function is not initialized.")
        kl_loss = self.loss_fn(f_pc_mean, f_pc_var, zeros, self.var) \
            + self.loss_fn(zeros, self.var, f_pc_mean, f_pc_var)

        kl_loss = (kl_loss * self.dim_weight).sum()

        return {"y_pred": y_pred, "feat_pc": f_pc}, kl_loss

    def _update(self, engine, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).float()
        x, y = _select_center_frame(x, y)

        loss: Tensor | None = None

        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

        output, loss = self.adapt_step(x, y) # type: ignore

        if loss is not None and self.opt is not None:
            loss.backward()
            self.opt.step()

        output["y"] = y # type: ignore
        return output

    def _get_regressor_weight(self) -> Tensor:
        regressor_module = getattr(self.net, "regressor", None)
        weight = getattr(regressor_module, "weight", None)
        if weight is None:
            raise AttributeError("Regressor must have a weight attribute.")
        return weight
