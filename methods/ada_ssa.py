import math
from dataclasses import InitVar, dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from methods.base import BaseTTA
from .utils import get_pca_basis


class TensorCallable(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


@dataclass
class ShiftDetector:
    t_threshold: float = 3.0

    @torch.no_grad()
    def detect_shift(self,
                     prev_feats: Tensor | None,
                     curr_feats: Tensor | None) -> bool:
        if prev_feats is None or curr_feats is None:
            return False

        m1 = prev_feats.mean(dim=0)
        m2 = curr_feats.mean(dim=0)
        v1 = prev_feats.var(dim=0, unbiased=True) + 1e-6
        v2 = curr_feats.var(dim=0, unbiased=True) + 1e-6

        n1 = max(prev_feats.size(0), 1)
        n2 = max(curr_feats.size(0), 1)

        se = torch.sqrt(v1 / n1 + v2 / n2)
        t_stat = (m1 - m2).abs() / (se + 1e-6)
        t_mean = t_stat.mean().item()
        t_max = t_stat.max().item()
        # print(f"ShiftDetector: t-statistic mean = {t_mean:.4f}")
        print(f"ShiftDetector: t-statistic max = {t_max:.4f}")
        return t_max > self.t_threshold


@dataclass
class GlobalStats:
    num_features: int
    ema_alpha: float = 0.1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.mu = torch.zeros(self.num_features, device=self.device)
        self.mu2 = torch.ones(self.num_features, device=self.device)
        self.initialized = False
        self.num_updates = 0

    @torch.no_grad()
    def update(self, batch_feats: Tensor, alpha: float | None = None) -> None:
        batch_mu = batch_feats.mean(dim=0)
        batch_var = batch_feats.var(dim=0, unbiased=True) + 1e-6
        batch_mu2 = batch_var + batch_mu.pow(2)

        if not self.initialized:
            self.mu.copy_(batch_mu)
            self.mu2.copy_(batch_mu2)
            self.initialized = True
            return

        # alpha = self.ema_alpha
        alpha = self.ema_alpha if alpha is None else alpha

        # self.num_updates += 1
        # alpha = 1.0 / self.num_updates

        self.mu = (1 - alpha) * self.mu + alpha * batch_mu
        self.mu2 = (1 - alpha) * self.mu2 + alpha * batch_mu2




    @property
    def var(self) -> Tensor:
        return torch.clamp(self.mu2 - self.mu.pow(2), min=1e-6)


class SSALoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                mu_a: Tensor,
                var_a: Tensor,
                mu_b: Tensor,
                var_b: Tensor) -> Tensor:
        var_a = var_a + 1e-6
        var_b = var_b + 1e-6
        diff = mu_a - mu_b
        kl_ab = 0.5 * (torch.log(var_b / var_a)
                       + (var_a + diff.pow(2)) / var_b
                       - 1.0)
        kl_ba = 0.5 * (torch.log(var_a / var_b)
                       + (var_b + diff.pow(2)) / var_a
                       - 1.0)
        return (kl_ab + kl_ba).mean()


@dataclass
class AdaptiveSSA(BaseTTA):
    pc_config: InitVar[dict | None] = None
    ema_alpha: float = 0.1
    base_ssa_weight: float = 0.0
    max_ssa_weight: float = 1.0
    ssa_growth_rate: float = 0.1
    t_threshold: float = 3.0
    alpha_batch: float = 0.5
    main_loss: str = "none"

    feature_extractor: TensorCallable | None = field(init=False, default=None, repr=False)
    predictor: TensorCallable | None = field(init=False, default=None, repr=False)
    shift_detector: ShiftDetector = field(init=False, repr=False)
    ssa_loss_fn: SSALoss = field(init=False, repr=False)
    global_stats: GlobalStats | None = field(init=False, default=None, repr=False)
    prev_batch_feats_for_shift: Tensor | None = field(init=False, default=None, repr=False)
    shift_count: int = field(init=False, default=0)
    feature_mean: Tensor | None = field(init=False, default=None, repr=False)
    pca_mean: Tensor | None = field(init=False, default=None, repr=False)
    pca_basis: Tensor | None = field(init=False, default=None, repr=False)
    pca_var: Tensor | None = field(init=False, default=None, repr=False)
    _pre_batch_mu: Tensor | None = field(init=False, default=None, repr=False)
    _pre_batch_mu2: Tensor | None = field(init=False, default=None, repr=False)
    source_sample_count: int | None = field(default=None, repr=False)

    mu_param_shifts: list = field(init=False, default_factory=list, repr=False)
    mu2_param_shifts: list = field(init=False, default_factory=list, repr=False)
    target_sample_count: int | None = field(default=None, repr=False)
    sample_alpha: float = field(default=1.0, repr=False)

    def __post_init__(self, compile_model: dict | None, pc_config: dict | None):
        self._pc_config = dict(pc_config or {})
        super().__post_init__(compile_model)
        self._init_subspace()
        feature = getattr(self.net, "feature", None)
        predictor = getattr(self.net, "predict_from_feature", None)
        if not callable(feature) or not callable(predictor):
            raise AttributeError("Net must define callable feature and predict_from_feature methods.")
        self.feature_extractor = cast(TensorCallable, feature)
        self.predictor = cast(TensorCallable, predictor)
        self.shift_detector = ShiftDetector(t_threshold=self.t_threshold)
        self.ssa_loss_fn = SSALoss()
        self.prev_batch_feats_for_shift = None
        self.shift_count = 1

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.feature_extractor is None or self.predictor is None:
            raise RuntimeError("AdaptiveSSA feature extractor is not initialized.")

        features = self.feature_extractor(x)
        flat_feat = features.reshape(features.size(0), -1)
        y_pred = self.predictor(flat_feat).view(-1)
        feats_subspace = self._project_to_subspace(flat_feat)

        if self.global_stats is None:
            raise RuntimeError("Global statistics are not initialized for AdaptiveSSA.")
        if self.pca_mean is None or self.pca_var is None:
            raise RuntimeError("PCA statistics are not initialized for AdaptiveSSA.")

        prev_global_mu = self.global_stats.mu.detach()
        prev_global_mu2 = self.global_stats.mu2.detach()
        batch_mu = feats_subspace.mean(dim=0)
        batch_var = feats_subspace.var(dim=0, unbiased=True) + 1e-6
        batch_mu2 = batch_var + batch_mu.pow(2)
        self._pre_batch_mu = batch_mu.detach()
        self._pre_batch_mu2 = batch_mu2.detach()
        global_mu = self.alpha_batch * batch_mu + (1 - self.alpha_batch) * prev_global_mu
        global_mu2 = self.alpha_batch * batch_mu2 + (1 - self.alpha_batch) * prev_global_mu2

        # if len(self.mu_param_shifts) > 0:
        #     shift_mu = torch.stack(self.mu_param_shifts).sum(dim=0)
        #     shift_mu2 = torch.stack(self.mu2_param_shifts).sum(dim=0)

        #     print(f"shift_mu norm: {shift_mu.norm().item():.6f}, shift_mu2 norm: {shift_mu2.norm().item():.6f}")

        #     global_mu += shift_mu.to(global_mu.device) * self.sample_alpha
        #     global_mu2 += shift_mu2.to(global_mu2.device) * self.sample_alpha


        global_var = torch.clamp(global_mu2 - global_mu.pow(2), min=1e-6)
        # global_var = self.alpha_batch * batch_var + (1 - self.alpha_batch) * (prev_global_mu2 - prev_global_mu.pow(2))

        ## No weighting based on shift detection
        ssa_weight = self.max_ssa_weight
        shifted = False
        ssa_loss = self.ssa_loss_fn(global_mu, global_var, self.pca_mean, self.pca_var)
        total_loss = float(ssa_weight) * ssa_loss

        ## Schedule SSA weight based on detected shifts
        # ssa_weight = self._compute_ssa_weight()
        # ssa_loss = self.ssa_loss_fn(global_mu, global_var, self.pca_mean, self.pca_var)
        # total_loss = float(ssa_weight) * ssa_loss

        # self.global_stats.update(feats_subspace.detach())

        # shifted = self.shift_detector.detect_shift(self.prev_batch_feats_for_shift, feats_subspace.detach())
        # if shifted:
        #     self.shift_count += 1
        # self.prev_batch_feats_for_shift = feats_subspace.detach()

        output = {
            "y_pred": y_pred,
            "ssa_weight": torch.tensor(ssa_weight, device=x.device),
            "ssa_loss": ssa_loss.detach(),
            "shifted": torch.tensor(1.0 if shifted else 0.0, device=x.device),
            "shift_count": torch.tensor(float(self.shift_count), device=x.device),
        }
        return output, total_loss

    def _compute_ssa_weight(self) -> float:
        w = self.max_ssa_weight * (1.0 - math.exp(-self.ssa_growth_rate * self.shift_count))
        w = self.base_ssa_weight + w
        print(f"AdaptiveSSA: SSA weight = {w:.4f}")
        return min(w, self.max_ssa_weight)

    def _init_subspace(self) -> None:
        if not self._pc_config:
            raise ValueError("pc_config must be provided for AdaptiveSSA to define the PCA subspace.")
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.feature_mean = mean.to(self.device)
        self.pca_basis = basis.to(self.device)
        self.pca_var = var.to(self.device)
        self.pca_mean = torch.zeros_like(self.pca_var)
        stats = GlobalStats(
            num_features=self.pca_basis.shape[1],
            ema_alpha=self.ema_alpha,
            device=self.device,
        )
        stats.mu.zero_()
        stats.mu2.copy_(self.pca_var.clone())
        stats.initialized = True # Start with PCA stats
        self.global_stats = stats
        # if self.source_sample_count is None:
        #     src_count = self._pc_config.get("source_sample_count") or self._pc_config.get("num_source_samples")
        #     if src_count is not None:
        #         self.source_sample_count = int(src_count)
        if self.source_sample_count is None:
            self.source_sample_count = self._infer_source_count_from_stat_file()
            print(f"AdaptiveSSA: inferred source sample count = {self.source_sample_count}")

    def _project_to_subspace(self, flat_feat: Tensor) -> Tensor:
        if self.feature_mean is None or self.pca_basis is None:
            raise RuntimeError("PCA subspace is not initialized.")
        return (flat_feat - self.feature_mean) @ self.pca_basis

    def _update(self, engine, batch):
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).float().flatten()

        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

        output, loss = self.adapt_step(x, y)

        if loss is not None and self.opt is not None:
            loss.backward()
            self.opt.step()
            if self._pre_batch_mu is not None and self._pre_batch_mu2 is not None:
                post_mu, post_mu2, post_feats = self._compute_batch_stats(x)
                self._apply_stat_shift(post_mu, post_mu2)
                if self.global_stats is not None:
                    # self.global_stats.update(post_feats.detach())
                    alpha = self._get_alpha_for_update(post_feats)
                    self.sample_alpha = alpha if alpha is not None else self.sample_alpha
                    self.global_stats.update(post_feats.detach())
                    if self.source_sample_count is not None:
                        self.source_sample_count += post_feats.size(0)

        output["y"] = y
        return output

    @torch.no_grad()
    def _compute_batch_stats(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.feature_extractor is None:
            raise RuntimeError("AdaptiveSSA feature extractor is not initialized.")
        feats = self.feature_extractor(x)
        flat_feat = feats.reshape(feats.size(0), -1)
        feats_subspace = self._project_to_subspace(flat_feat)
        mu = feats_subspace.mean(dim=0)
        var = feats_subspace.var(dim=0, unbiased=True) + 1e-6
        mu2 = var + mu.pow(2)
        return mu, mu2, feats_subspace

    @torch.no_grad()
    def _apply_stat_shift(self, post_mu: Tensor, post_mu2: Tensor) -> None:
        if (self.global_stats is None or self._pre_batch_mu is None
                or self._pre_batch_mu2 is None):
            self._pre_batch_mu = None
            self._pre_batch_mu2 = None
            return
        stats = self.global_stats

        shift_mu = post_mu - self._pre_batch_mu
        shift_mu2 = post_mu2 - self._pre_batch_mu2

        self.mu_param_shifts.append(shift_mu.cpu())
        self.mu2_param_shifts.append(shift_mu2.cpu())

        mu_new = stats.mu
        mu2_new = stats.mu2
        var_new = torch.clamp(mu2_new - mu_new.pow(2), min=1e-6)
        stats.mu.copy_(mu_new)
        stats.mu2.copy_(var_new + mu_new.pow(2))
        self._pre_batch_mu = None
        self._pre_batch_mu2 = None

    def _get_alpha_for_update(self, batch_feats: Tensor) -> float | None:
        if self.source_sample_count is None:
            return None
        src_count = float(max(self.source_sample_count, 1))
        return float(batch_feats.size(0)) / src_count

    def _infer_source_count_from_stat_file(self) -> int | None:
        print("AdaptiveSSA: inferring source sample count from stat_file...")
        print(self._pc_config)
        stat_file = self._pc_config.get("stat_file")
        feature_file = stat_file.replace("stats", "valid_features") if stat_file else None
        print(f"AdaptiveSSA: feature_file = {feature_file}")
        if not feature_file:
            return None
        try:
            data = torch.load(feature_file, map_location="cpu")
        except (FileNotFoundError, RuntimeError, OSError) as err:
            print(f"AdaptiveSSA: failed to load feature_file for source count: {err}")
            return None

        if isinstance(data, torch.Tensor):
            if data.dim() == 2 and data.size(1) >= 2:
                return int(data.size(0))
            return None

        if isinstance(data, dict):
            for key in ("feat_labels", "features"):
                tensor = data.get(key)
                if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
                    return int(tensor.size(0))
        return None

