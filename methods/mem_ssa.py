from dataclasses import InitVar, dataclass, field
import copy

import torch
from torch import Tensor, nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss


@torch.no_grad()
def ema_update(model: nn.Module, source: nn.Module, momentum: float) -> None:
    """Updates model parameters in-place using EMA."""
    if momentum >= 1.0:
        return
    for p, src_p in zip(model.parameters(), source.parameters()):
        if p.requires_grad:
            p.data.lerp_(src_p.data, weight=1.0 - momentum)


def debug_tensor(name: str, t: Tensor) -> None:
    """Print debug info for a tensor."""
    if torch.isnan(t).any():
        print(f"[DEBUG] {name}: contains NaN")
    if torch.isinf(t).any():
        print(f"[DEBUG] {name}: contains Inf")
    print(f"[DEBUG] {name}: min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}")


@dataclass
class StatsMemory:
    """FIFO buffer for batch statistics (MemBN style)."""
    buffer_size: int
    num_features: int
    gamma: float = 0.3
    min_alpha: float = 0.5  # Minimum source weight to prevent collapse
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.mu_buf = torch.zeros(self.buffer_size, self.num_features, device=self.device)
        self.var_buf = torch.ones(self.buffer_size, self.num_features, device=self.device)
        self.ptr = 0
        self.count = 0

    def update(self, mu: Tensor, var: Tensor) -> None:
        """FIFO update (detached)."""
        self.mu_buf[self.ptr] = mu.detach()
        self.var_buf[self.ptr] = var.detach().clamp(min=1e-6)
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    def get_mixed_stats(self, mu_in: Tensor, var_in: Tensor, 
                        src_mu: Tensor, src_var: Tensor) -> tuple[Tensor, Tensor, float]:
        """Compute MemBN-style mixed statistics. Returns (mu_mix, var_mix, alpha)."""
        var_in = var_in.clamp(min=1e-6)
        src_var = src_var.clamp(min=1e-6)
        
        # Early stage: use source statistics more heavily
        if self.count < 10:
            alpha = 0.9  # Start with high source weight
            mu_mix = alpha * src_mu + (1 - alpha) * mu_in
            var_mix = alpha * src_var + (1 - alpha) * var_in
            return mu_mix, var_mix.clamp(min=1e-6), alpha
        
        # Compute memory statistics
        hist_mu = self.mu_buf[:self.count]
        hist_var = self.var_buf[:self.count].clamp(min=1e-6)
        all_mu = torch.cat([mu_in.unsqueeze(0), hist_mu], dim=0)
        all_var = torch.cat([var_in.unsqueeze(0), hist_var], dim=0)

        mu_mem = all_mu.mean(dim=0)
        var_mem = all_var.mean(dim=0) + ((all_mu - mu_mem.unsqueeze(0)) ** 2).mean(dim=0)
        var_mem = var_mem.clamp(min=1e-6)

        # Compute alpha using normalized KL divergence
        # Normalize by source variance to handle scale differences
        kl_to_src = self._normalized_kl(mu_in, var_in, src_mu, src_var)
        kl_to_mem = self._normalized_kl(mu_in, var_in, mu_mem, var_mem)
        
        kl_sum = kl_to_src + kl_to_mem + 1e-8
        
        # alpha = weight for source (higher when closer to source)
        raw_alpha = kl_to_mem / kl_sum  # Close to source -> high alpha
        alpha = max(self.min_alpha, min(0.95, raw_alpha.item())) ** self.gamma

        mu_mix = alpha * src_mu + (1 - alpha) * mu_mem
        var_mix = (alpha * src_var + (1 - alpha) * var_mem + 
                   alpha * (1 - alpha) * (src_mu - mu_mem) ** 2)
        var_mix = var_mix.clamp(min=1e-6)

        return mu_mix, var_mix, alpha

    def _normalized_kl(self, mu1: Tensor, var1: Tensor, mu2: Tensor, var2: Tensor) -> Tensor:
        """Compute normalized KL divergence (scale-invariant)."""
        # Normalize variances to handle large scale differences
        var1_norm = var1 / (var2 + 1e-6)
        diff_norm = (mu1 - mu2) ** 2 / (var2 + 1e-6)
        kl = 0.5 * (var1_norm + diff_norm - 1 - var1_norm.log()).sum()
        return kl.clamp(min=0.0, max=1000.0)


@dataclass
class Mem_SSA(BaseTTA):
    """MemBN-style SSA with statistics memory buffer."""
    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None

    buffer_size: int = 512
    min_buffer_size: int = 1
    stats_buffer_size: int = 128
    membn_gamma: float = 0.5  # Reduced from 0.3 for more conservative mixing
    min_source_weight: float = 0.5  # Minimum weight for source statistics

    ssa_weight: float = 1.0
    ema_momentum: float = 0.999  # Higher momentum = more conservative
    weight_bias: float = 1.0
    weight_exp: float = 0.5  # Reduced to dampen large weight differences
    
    debug: bool = False

    stats_memory: StatsMemory | None = field(init=False, default=None)
    feature_mean: Tensor | None = field(init=False, default=None)
    pca_basis: Tensor | None = field(init=False, default=None)
    pca_mean: Tensor | None = field(init=False, default=None)
    pca_var: Tensor | None = field(init=False, default=None)
    dim_weight: Tensor | None = field(init=False, default=None)
    buffer_imgs: Tensor | None = field(init=False, default=None)
    source_net: nn.Module | None = field(init=False, default=None)
    _debug_count: int = field(init=False, default=0)

    def __post_init__(self, compile_model, val_dataset, target_names, pc_config, loss_config):
        self._pc_config = pc_config or {}
        self._loss_config = loss_config or {}
        super().__post_init__(compile_model, val_dataset, target_names)

        # Init PCA
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.feature_mean = mean.to(self.device)
        self.pca_basis = basis.to(self.device)
        self.pca_var = var.to(self.device).clamp(min=1e-6)
        self.pca_mean = torch.zeros_like(self.pca_var)
        
        if self.debug:
            print(f"[INIT] pca_var: min={self.pca_var.min():.6f}, max={self.pca_var.max():.6f}")
            print(f"[INIT] feature_mean: min={self.feature_mean.min():.6f}, max={self.feature_mean.max():.6f}")

        num_features = self.pca_basis.shape[1]
        self.stats_memory = StatsMemory(
            self.stats_buffer_size, num_features, self.membn_gamma, 
            self.min_source_weight, self.device
        )

        # Dimension weighting with variance normalization
        if hasattr(self.net, "regressor"):
            with torch.no_grad():
                w = self.net.regressor.weight  # type: ignore
                dim_w = torch.abs(w @ self.pca_basis).sum(dim=0)  # type: ignore
                # Normalize by variance to handle scale differences
                dim_w = dim_w / (torch.sqrt(self.pca_var) + 1e-6)
                self.dim_weight = (dim_w + self.weight_bias).pow(self.weight_exp)
                if self.debug:
                    print(f"[INIT] dim_weight: min={self.dim_weight.min():.6f}, max={self.dim_weight.max():.6f}")
        else:
            self.dim_weight = torch.ones(num_features, device=self.device)

        self.source_net = copy.deepcopy(self.net).to(self.device).requires_grad_(False)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        self._debug_count += 1
        do_debug = self.debug and self._debug_count <= 3
        
        # 1. Forward
        feats = self.net.feature(x)  # type: ignore
        flat_feats = feats.flatten(1)
        
        if do_debug:
            debug_tensor("flat_feats", flat_feats)
        
        y_pred = self.net.predict_from_feature(flat_feats)  # type: ignore

        # 2. Project to subspace
        centered = flat_feats - self.feature_mean
        projected = centered @ self.pca_basis
        
        if do_debug:
            debug_tensor("projected", projected)

        # 3. Batch statistics (normalized by source variance)
        mu_in = projected.mean(dim=0)
        if projected.shape[0] > 1:
            var_in = projected.var(dim=0, unbiased=False).clamp(min=1e-6)
        else:
            var_in = self.pca_var.clone()  # type: ignore # Use source variance for single sample
        
        if do_debug:
            debug_tensor("mu_in", mu_in)
            debug_tensor("var_in", var_in)

        # 4. Get mixed statistics (MemBN)
        mu_mix, var_mix, alpha = self.stats_memory.get_mixed_stats(  # type: ignore
            mu_in, var_in, self.pca_mean, self.pca_var  # type: ignore
        )
        
        if do_debug:
            print(f"[DEBUG] alpha: {alpha:.4f}")
            debug_tensor("mu_mix", mu_mix)
            debug_tensor("var_mix", var_mix)

        # 5. Symmetric KL loss (normalized)
        var_mix_safe = var_mix.clamp(min=1e-6)
        pca_var_safe = self.pca_var.clamp(min=1e-6)  # type: ignore
        
        # Normalize KL by source variance for scale-invariance
        kl_fwd = diagonal_gaussian_kl_loss(
            mu_mix / torch.sqrt(pca_var_safe), var_mix_safe / pca_var_safe,
            self.pca_mean / torch.sqrt(pca_var_safe), torch.ones_like(pca_var_safe),  # type: ignore
            dim_reduction="none", **self._loss_config
        )
        kl_bwd = diagonal_gaussian_kl_loss(
            self.pca_mean / torch.sqrt(pca_var_safe), torch.ones_like(pca_var_safe),  # type: ignore
            mu_mix / torch.sqrt(pca_var_safe), var_mix_safe / pca_var_safe,
            dim_reduction="none", **self._loss_config
        )
        
        if do_debug:
            debug_tensor("kl_fwd", kl_fwd)
            debug_tensor("kl_bwd", kl_bwd)
        
        
        ssa_loss = ((kl_fwd + kl_bwd) * self.dim_weight).mean()  # type: ignore # Use mean instead of sum

        if torch.isnan(ssa_loss) or torch.isinf(ssa_loss):
            if do_debug:
                print(f"[DEBUG] ssa_loss is NaN/Inf, resetting to 0")
            ssa_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        # 6. Update buffers
        with torch.no_grad():
            self.stats_memory.update(mu_in, var_in)  # type: ignore
            buffer_ready = self._update_buffer(x)

        return {
            "y_pred": y_pred,
            "ssa_loss": ssa_loss.detach(),
            "alpha": torch.tensor(alpha, device=x.device),
            "buffer_ready": torch.tensor(float(buffer_ready), device=x.device)
        }, self.ssa_weight * ssa_loss

    def _update(self, engine, batch):
        self.net.train() if self.train_mode else self.net.eval()
        x, y = batch[0].to(self.device), batch[1].to(self.device).float()

        if self.opt:
            self.opt.zero_grad(set_to_none=True)

        out, loss = self.adapt_step(x, y)

        if self.opt and loss is not None and loss.requires_grad and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)  # Reduced from 1.0
            self.opt.step()
            ema_update(self.net, self.source_net, self.ema_momentum)  # type: ignore

        out["y"] = y
        return out

    @torch.no_grad()
    def _update_buffer(self, x: Tensor) -> bool:
        x_cpu = x.detach().cpu()

        if self.buffer_imgs is None:
            self.buffer_imgs = x_cpu
        elif self.buffer_imgs.size(0) < self.buffer_size:
            self.buffer_imgs = torch.cat([self.buffer_imgs, x_cpu])
        else:
            idx = torch.randint(0, self.buffer_size, (x_cpu.size(0),))
            self.buffer_imgs[idx] = x_cpu

        return self.buffer_imgs.size(0) >= self.min_buffer_size