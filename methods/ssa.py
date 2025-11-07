from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .base import TTAMethod


class SSAMethod(TTAMethod):
    """Significant Subspace Alignment for regression."""

    def __init__(self, model: nn.Module, cfg: Mapping[str, Any], device: torch.device) -> None:
        super().__init__(model=model, cfg=cfg, device=device, name="ssa")
        self.subspace_dim = int(cfg.get("subspace_dim", 16))
        self.adapt_lr = float(cfg.get("adapt_lr", 5e-2))
        self.regularizer = float(cfg.get("regularizer", 1e-2))
        self.stats_momentum = float(cfg.get("stats_momentum", 0.1))
        self.adapter: Optional[nn.Linear] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._feature_dim: Optional[int] = None
        self._source_mean: Optional[torch.Tensor] = None
        self._source_basis: Optional[torch.Tensor] = None
        self._source_mean_coeffs: Optional[torch.Tensor] = None
        self._identity: Optional[torch.Tensor] = None
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def set_source_statistics(self, stats: Mapping[str, torch.Tensor]) -> None:
        mean = stats.get("mean")
        cov = stats.get("cov")
        if mean is None or cov is None:
            raise ValueError("SSA requires both mean and covariance statistics.")
        mean = mean.to(self.device)
        cov = cov.to(self.device)
        feature_dim = int(mean.numel())
        k = min(self.subspace_dim, feature_dim)
        evals, evecs = torch.linalg.eigh(cov)
        topk = torch.topk(evals, k=k, largest=True)
        basis = evecs[:, topk.indices]
        self._feature_dim = feature_dim
        self._source_mean = mean
        self._source_basis = basis
        self._source_mean_coeffs = mean @ basis
        self.adapter = nn.Linear(k, k, bias=False)
        self.adapter.to(self.device)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(k, device=self.device))
        self._identity = torch.eye(k, device=self.device)
        self.optimizer = torch.optim.SGD(self.adapter.parameters(), lr=self.adapt_lr, momentum=0.0)

    def train_mode(self) -> None:
        self.model.eval()
        if self.adapter is not None:
            self.adapter.train()

    def eval_mode(self) -> None:
        self.model.eval()
        if self.adapter is not None:
            self.adapter.eval()

    def compute_subspace(self) -> torch.Tensor:
        if self._source_basis is None:
            raise RuntimeError("SSA subspace not initialised. Call set_source_statistics first.")
        return self._source_basis

    def project_to_subspace(self, features: torch.Tensor) -> torch.Tensor:
        basis = self.compute_subspace()
        return features @ basis

    def _reconstruct_from_subspace(self, coefficients: torch.Tensor) -> torch.Tensor:
        basis = self.compute_subspace()
        return coefficients @ basis.T

    def align_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            return features
        coeffs = self.project_to_subspace(features)
        adapted_coeffs = self.adapter(coeffs)
        projected = self._reconstruct_from_subspace(adapted_coeffs)
        residual = features - self._reconstruct_from_subspace(coeffs)
        return projected + residual

    def adapt_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        if self.adapter is None or self.optimizer is None or self._source_mean_coeffs is None:
            raise RuntimeError("SSA is not initialised. Ensure source statistics are provided.")
        start = time.perf_counter()
        inputs = batch["inputs"].to(self.device)
        with torch.no_grad():
            _, features = self.model(inputs, return_features=True)
        coeffs = self.project_to_subspace(features.detach())
        self.optimizer.zero_grad(set_to_none=True)
        adapted = self.adapter(coeffs)
        alignment_loss = F.mse_loss(adapted.mean(dim=0), self._source_mean_coeffs)
        reg_loss = F.mse_loss(self.adapter.weight, self._identity)
        loss = alignment_loss + self.regularizer * reg_loss
        loss.backward()
        self.optimizer.step()
        elapsed = time.perf_counter() - start
        return {
            "loss_proxy": float(loss.item()),
            "alignment_loss": float(alignment_loss.item()),
            "regularizer": float(reg_loss.item()),
            "step_time": elapsed,
        }

    def predict(self, batch: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if self._source_basis is None:
            raise RuntimeError("SSA predict called before initialisation.")
        inputs = batch["inputs"] if isinstance(batch, Mapping) else batch
        inputs = inputs.to(self.device)
        with torch.no_grad():
            _, features = self.model(inputs, return_features=True)
            aligned = self.align_features(features)
            outputs = self.model.forward_head(aligned)
        return outputs

    def state_dict(self) -> Mapping[str, Any]:
        if self.adapter is None:
            return {}
        return {"adapter": self.adapter.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not state_dict:
            return
        if self.adapter is None:
            raise RuntimeError("SSA adapter not initialised. Load source statistics before state dict.")
        adapter_state = state_dict.get("adapter")
        if adapter_state is not None:
            self.adapter.load_state_dict(adapter_state)