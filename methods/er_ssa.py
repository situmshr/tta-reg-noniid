from dataclasses import InitVar, dataclass, field
from typing import Callable, cast
import copy

import torch
from torch import Tensor, nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss


@torch.no_grad()
def ema_update(model: nn.Module, source: nn.Module, momentum: float) -> None:
    """Updates model parameters in-place using EMA."""
    if momentum >= 1.0: return
    for p, src_p in zip(model.parameters(), source.parameters()):
        if p.requires_grad:
            p.data.lerp_(src_p.data, weight=1.0 - momentum)


import torch
from torch import Tensor
from dataclasses import dataclass, field

@dataclass
class GlobalStats:
    """Manages global running statistics using Mean and Variance directly."""
    num_features: int
    alpha: float = 0.1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.mu = torch.zeros(self.num_features, device=self.device)
        self.var = torch.ones(self.num_features, device=self.device)
        self.initialized = False
        self.stat_update = False

    def update(self, batch_mu: Tensor, batch_var: Tensor) -> tuple[Tensor, Tensor]:
        """
        Updates running statistics using the logic from Eq. (6) in the paper.
        Combines previous stats and batch stats as a mixture of Gaussians.
        """
        if not self.initialized:
            self.mu.data.copy_(batch_mu.detach())
            self.var.data.copy_(batch_var.detach())
            self.initialized = True
            return batch_mu, batch_var

        prev_mu = self.mu.detach()
        prev_var = self.var.detach()

        # mu_new = (1 - alpha) * mu_prev + alpha * mu_batch
        new_mu = torch.lerp(prev_mu, batch_mu, self.alpha)

        # var_new = (1-alpha)*var_prev + alpha*var_batch + alpha*(1-alpha)*(mu_prev - mu_batch)^2
        
        # alpha * sigma_s^2 + (1-alpha) * sigma_t^2
        var_linear = torch.lerp(prev_var, batch_var, self.alpha)
        
        # alpha * (1-alpha) * (mu_s - mu_t)^2
        mean_diff_term = self.alpha * (1.0 - self.alpha) * (prev_mu - batch_mu).pow(2)
        
        new_var = var_linear + mean_diff_term

        if self.stat_update:
            self.mu.data.copy_(new_mu.detach())
            self.var.data.copy_(new_var.detach())
        
        return new_mu, new_var


@dataclass
class ER_SSA(BaseTTA):
    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None

    buffer_size: int = 512
    min_buffer_size: int = 1
    ssa_weight: float = 1.0
    ema_alpha: float = 1.0
    ema_momentum: float = 0.99
    weight_bias: float = 1.0
    weight_exp: float = 1.0

    # Components
    global_stats: GlobalStats | None = field(init=False, default=None)
    feature_mean: Tensor | None = field(init=False, default=None)
    pca_basis: Tensor | None = field(init=False, default=None)
    pca_mean: Tensor | None = field(init=False, default=None)
    pca_var: Tensor | None = field(init=False, default=None)
    dim_weight: Tensor | None = field(init=False, default=None)
    
    # GDM Buffer (CPU)
    buffer_imgs: Tensor | None = field(init=False, default=None)
    buffer_scores: Tensor | None = field(init=False, default=None)
    source_net: nn.Module | None = field(init=False, default=None)

    def __post_init__(self, compile_model, val_dataset, target_names, pc_config, loss_config):
        self._pc_config = pc_config or {}
        self._loss_config = loss_config or {}
        super().__post_init__(compile_model, val_dataset, target_names)

        # Init PCA & Stats
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.feature_mean = mean.to(self.device)
        self.pca_basis = basis.to(self.device)
        self.pca_var = var.to(self.device)
        self.pca_mean = torch.zeros_like(self.pca_var)
        self.global_stats = GlobalStats(self.pca_basis.shape[1], self.ema_alpha, self.device)

        with torch.no_grad():
            self.global_stats.mu.copy_(self.pca_mean)
            self.global_stats.var.copy_(self.pca_var)
            self.global_stats.initialized = True

        # Init Weights
        if hasattr(self.net, "regressor"):
            with torch.no_grad():
                w = self.net.regressor.weight # type: ignore
                dim_w = torch.abs(w @ self.pca_basis).sum(dim=0) # type: ignore
                self.dim_weight = (dim_w + self.weight_bias).pow(self.weight_exp)

        # Init EMA Source
        self.source_net = copy.deepcopy(self.net).to(self.device).requires_grad_(False)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        # 1. Forward & Buffer Logic
        use_buffer = (self.buffer_imgs is not None) and (self.buffer_imgs.size(0) >= self.min_buffer_size)

        if not use_buffer:
            feats = self.net.feature(x) # type: ignore
            feats_stats = feats
        else:
            # Sample from buffer
            n_sample = x.shape[0]
            idxs = torch.randperm(self.buffer_imgs.size(0))[:n_sample] # type: ignore
            x_buf = self.buffer_imgs[idxs].to(self.device) # type: ignore
            
            # Combined forward for efficiency
            feats_all = self.net.feature(torch.cat([x, x_buf])) # type: ignore
            feats = feats_all[:n_sample]
            feats_stats = feats_all
            self.global_stats.stat_update = True # type: ignore

        # 2. Prediction
        y_pred = self.net.predict_from_feature(feats) # type: ignore

        # 3. Update Stats & Calculate Loss
        # Inline projection: (feats - mean) @ basis
        projected = (feats_stats - self.feature_mean) @ self.pca_basis

        # Single update call
        g_mu, g_var = self.global_stats.update( # type: ignore
            projected.mean(dim=0), 
            projected.var(dim=0, unbiased=False)
        )

        # Symmetric KL
        kl_fwd = diagonal_gaussian_kl_loss(g_mu, g_var, self.pca_mean, self.pca_var, dim_reduction="none", **self._loss_config) # type: ignore
        kl_bwd = diagonal_gaussian_kl_loss(self.pca_mean, self.pca_var, g_mu, g_var, dim_reduction="none", **self._loss_config) # type: ignore
        ssa_loss = ((kl_fwd + kl_bwd) * self.dim_weight).sum() # type: ignore

        # if use_buffer:
        #     with torch.no_grad():
        #         self.global_stats.stat_update = True # type: ignore
        #         buffer_projected = projected[n_sample:] # type: ignore
        #         self.global_stats.update( # type: ignore
        #             buffer_projected.mean(dim=0), 
        #             buffer_projected.var(dim=0, unbiased=False)
        #         )
        #         self.global_stats.stat_update = False # type: ignore

        # 4. Update Buffer (GDM)
        self._update_buffer(x)

        return {
            "y_pred": y_pred,
            "ssa_loss": ssa_loss.detach(),
            "buffer_ready": torch.tensor(float(use_buffer), device=x.device)
        }, self.ssa_weight * ssa_loss

    def _update(self, engine, batch):
        self.net.train() if self.train_mode else self.net.eval()
        x, y = batch[0].to(self.device), batch[1].to(self.device).float()

        if self.opt: self.opt.zero_grad(set_to_none=True)
        
        out, loss = self.adapt_step(x, y)

        if self.opt and loss is not None:
            loss.backward()
            self.opt.step()
            ema_update(self.net, self.source_net, self.ema_momentum) # type: ignore

        out["y"] = y
        return out

    @torch.no_grad()
    def _update_buffer(self, x: Tensor):
        x_cpu = x.detach().cpu()
        scores = torch.randn(x_cpu.shape[0]) # Random score for GDM

        if self.buffer_imgs is None:
            self.buffer_imgs, self.buffer_scores = x_cpu, scores
        else:
            self.buffer_imgs = torch.cat([self.buffer_imgs, x_cpu])
            self.buffer_scores = torch.cat([self.buffer_scores, scores]) # type: ignore

        if self.buffer_imgs.size(0) > self.buffer_size:
            # Keep top-k scores
            _, idx = torch.topk(self.buffer_scores, k=self.buffer_size, sorted=False)
            self.buffer_imgs = self.buffer_imgs[idx]
            self.buffer_scores = self.buffer_scores[idx]