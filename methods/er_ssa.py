from dataclasses import InitVar, dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss


class TensorCallable(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class KLCallable(Protocol):
    def __call__(self, m1: Tensor, v1: Tensor, m2: Tensor, v2: Tensor) -> Tensor: ...


@dataclass
class GlobalStats:
    num_features: int
    ema_alpha: float = 0.1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.mu = torch.zeros(self.num_features, device=self.device)
        self.mu2 = torch.ones(self.num_features, device=self.device)
        self.var = torch.ones(self.num_features, device=self.device)
        self.initialized = False

    def combine(self, batch_mu: Tensor, batch_var: Tensor, alpha: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if not self.initialized:
            raise RuntimeError("GlobalStats must be initialized before combining moments.")
        alpha = self.ema_alpha if alpha is None else alpha
        mu = (1 - alpha) * self.mu + alpha * batch_mu
        batch_mu2 = batch_var + batch_mu.pow(2)
        mu2 = (1 - alpha) * self.mu2 + alpha * batch_mu2
        var = torch.clamp(mu2 - mu.pow(2), min=1e-6)
        # var = (1 - alpha) * self.var + alpha * batch_var
        # mu2 = torch.clamp(var + mu.pow(2), min=1e-6)
        return mu, mu2, var

    @torch.no_grad()
    def commit(self, mu: Tensor, var: Tensor) -> None:
        self.mu.copy_(mu.detach())
        self.var.copy_(var.detach())
        self.mu2.copy_(torch.clamp(var + mu.pow(2), min=1e-6))
        self.initialized = True


class SSALoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        mu_a: Tensor,
        var_a: Tensor,
        mu_b: Tensor,
        var_b: Tensor,
    ) -> Tensor:
        var_a = var_a + 1e-6
        var_b = var_b + 1e-6
        diff = mu_a - mu_b
        kl_ab = 0.5 * (torch.log(var_b / var_a)
                       + (var_a + diff.pow(2)) / var_b
                       - 1.0)
        kl_ba = 0.5 * (torch.log(var_a / var_b)
                       + (var_b + diff.pow(2)) / var_a
                       - 1.0)
        return (kl_ab + kl_ba).sum()


@dataclass
class ER_SSA(BaseTTA):
    """
    Global Distribution Matching (GDM) を採用した ER_SSA。
    論文: "Selective Experience Replay for Lifelong Learning" (arXiv:1802.10269)
    """
    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None

    buffer_size: int = 512
    min_buffer_size: int = 1
    ssa_weight: float = 1.0
    ema_alpha: float = 0.1
    main_loss: str = "none"
    weight_bias: float = 1.0
    weight_exp: float = 1.0

    feature_extractor: TensorCallable | None = field(init=False, default=None, repr=False)
    predictor: TensorCallable | None = field(init=False, default=None, repr=False)
    loss_fn: KLCallable | None = field(init=False, default=None, repr=False)
    dim_weight: Tensor | None = field(init=False, default=None, repr=False)

    feature_mean: Tensor | None = field(init=False, default=None, repr=False)
    pca_basis: Tensor | None = field(init=False, default=None, repr=False)
    pca_mean: Tensor | None = field(init=False, default=None, repr=False)
    pca_var: Tensor | None = field(init=False, default=None, repr=False)
    global_stats: GlobalStats | None = field(init=False, default=None, repr=False)

    # GDM用バッファ: 画像データとそのスコア(乱数)を保持する
    # CPU上に保持してVRAMを節約する
    buffer_imgs: Tensor | None = field(init=False, default=None, repr=False)
    buffer_scores: Tensor | None = field(init=False, default=None, repr=False)
    buffer_sample_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self, compile_model: dict | None, pc_config: dict | None, loss_config: dict | None):
        self._pc_config = dict(pc_config or {})
        self._loss_config = dict(loss_config or {})
        super().__post_init__(compile_model)

        if self.min_buffer_size < 1:
            raise ValueError("min_buffer_size must be >= 1.")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError("min_buffer_size cannot exceed buffer_size.")

        self._init_subspace()

        feature = getattr(self.net, "feature", None)
        predictor = getattr(self.net, "predict_from_feature", None)
        if not callable(feature) or not callable(predictor):
            raise AttributeError("Net must define callable feature and predict_from_feature methods.")
        self.feature_extractor = cast(TensorCallable, feature)
        self.predictor = cast(TensorCallable, predictor)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.feature_extractor is None or self.predictor is None:
            raise RuntimeError("ER_SSA feature extractor is not initialized.")

        ssa_loss: Tensor = torch.tensor(0.0, device=x.device)
        total_loss: Tensor | None = None
        buffer_ready = False

        if self.pca_mean is None or self.pca_var is None:
            raise RuntimeError("PCA statistics are not initialized for ER_SSA.")
        if self.global_stats is None:
            raise RuntimeError("Global statistics are not initialized for ER_SSA.")
        if self.loss_fn is None or self.dim_weight is None:
            raise RuntimeError("ER_SSA loss function is not initialized.")

        if self.buffer_sample_count < self.min_buffer_size:
            # 現在バッチで通常の forward
            feats = self.feature_extractor(x)
            flat_feats = feats.reshape(feats.size(0), -1)
            y_pred = self.predictor(flat_feats).view(-1)

            # バッファ不足時は現在のバッチのみで統計を計算
            feats_subspace = self._project_to_subspace(flat_feats)
        else:
            # バッファに十分データがある場合は一様サンプリングしたバッファと元バッチを結合して統計を計算
            x_buf = self._sample_from_buffer(x.shape[0]).to(self.device)
            x_combined = torch.cat([x, x_buf], dim=0)
            feats = self.feature_extractor(x_combined)
            flat_feats_combined = feats.reshape(feats.size(0), -1)
            flat_feats = flat_feats_combined[:x.shape[0]]
            y_pred = self.predictor(flat_feats).view(-1)

            feats_subspace = self._project_to_subspace(flat_feats_combined)
            buffer_ready = True

        buf_mu = feats_subspace.mean(dim=0)
        buf_var = feats_subspace.var(dim=0, unbiased=True) + 1e-6

        global_mu, global_mu2, global_var = self.global_stats.combine(
            buf_mu, buf_var, alpha=self.ema_alpha
        )
        with torch.no_grad():
            self.global_stats.commit(global_mu, global_var)

        kl_forward = self.loss_fn(global_mu, global_var, self.pca_mean, self.pca_var)
        kl_backward = self.loss_fn(self.pca_mean, self.pca_var, global_mu, global_var)
        ssa_loss = (kl_forward + kl_backward) * self.dim_weight
        ssa_loss = ssa_loss.sum()
        total_loss = float(self.ssa_weight) * ssa_loss

        # Global Distribution Matching 戦略によるバッファ更新
        self._push_to_buffer(x)

        output = {
            "y_pred": y_pred,
            "ssa_loss": ssa_loss.detach(),
            "buffer_ready": torch.tensor(1.0 if buffer_ready else 0.0, device=x.device),
        }
        return output, total_loss

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

        if self.opt is not None and loss is not None:
            # print(f"ER_SSA_GDM loss: {loss.item():.6f}")
            loss.backward()
            self.opt.step()

        output["y"] = y
        return output

    def _init_subspace(self) -> None:
        if not self._pc_config:
            raise ValueError("pc_config must be provided for ER_SSA to define the PCA subspace.")
        mean, basis, var = get_pca_basis(**self._pc_config)
        self.feature_mean = mean.to(self.device)
        self.pca_basis = basis.to(self.device)
        self.pca_var = var.to(self.device)
        self.pca_mean = torch.zeros_like(self.pca_var)
        self.loss_fn = lambda m1, v1, m2, v2: diagonal_gaussian_kl_loss(
            m1, v1, m2, v2, dim_reduction="none", **self._loss_config
        )
        stats = GlobalStats(
            num_features=self.pca_basis.shape[1],
            ema_alpha=self.ema_alpha,
            device=self.device,
        )
        stats.commit(self.pca_mean, self.pca_var)
        self.global_stats = stats

        with torch.no_grad():
            regressor_weight = self._get_regressor_weight()
            dim_weight = torch.abs(regressor_weight @ self.pca_basis).flatten()
            self.dim_weight = (dim_weight + self.weight_bias).pow(self.weight_exp)
            print(f"ER_SSA dim_weight: {self.dim_weight}")

    def _project_to_subspace(self, flat_feats: Tensor) -> Tensor:
        if self.feature_mean is None or self.pca_basis is None:
            raise RuntimeError("PCA subspace is not initialized.")
        return (flat_feats - self.feature_mean) @ self.pca_basis

    def _compute_batch_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.feature_extractor is None:
            raise RuntimeError("ER_SSA feature extractor is not initialized.")
        feats = self.feature_extractor(x)
        flat_feats = feats.reshape(feats.size(0), -1)
        feats_subspace = self._project_to_subspace(flat_feats)
        mu = feats_subspace.mean(dim=0)
        var = feats_subspace.var(dim=0, unbiased=True) + 1e-6
        return mu, var
    
    def _get_regressor_weight(self) -> Tensor:
        regressor_module = getattr(self.net, "regressor", None)
        if regressor_module is None or not isinstance(regressor_module, nn.Module):
            raise AttributeError("Net must expose a regressor module with weight parameter.")
        weight = getattr(regressor_module, "weight", None)
        if not isinstance(weight, Tensor):
            raise AttributeError("Regressor module is missing weight tensor.")
        return weight

    @torch.no_grad()
    def _push_to_buffer(self, x: Tensor) -> None:
        """
        Global Distribution Matching (Reservoir Sampling) によるバッファ更新。
        論文 Eq(6): R(e_i) ~ N(0, 1) を割り当て、スコアが高いものを保持する。
        """
        x_cpu = x.detach().cpu()
        bsz = x_cpu.shape[0]

        # 新しいサンプルに対するランダムスコア生成 (Eq. 6)
        new_scores = torch.randn(bsz)

        if self.buffer_imgs is None or self.buffer_scores is None:
            # 初回
            self.buffer_imgs = x_cpu
            self.buffer_scores = new_scores
        else:
            # 既存バッファと結合
            self.buffer_imgs = torch.cat([self.buffer_imgs, x_cpu], dim=0)
            self.buffer_scores = torch.cat([self.buffer_scores, new_scores], dim=0)

        # バッファサイズ超過時の選定処理 (Distribution Matching)
        current_size = self.buffer_scores.size(0)
        if current_size > self.buffer_size:
            # スコアが高い順に上位 buffer_size 個を取得
            _, top_indices = torch.topk(self.buffer_scores, k=self.buffer_size, sorted=False)
            
            # 選択されたインデックスのみを残す
            self.buffer_imgs = self.buffer_imgs[top_indices]
            self.buffer_scores = self.buffer_scores[top_indices]

        self.buffer_sample_count = self.buffer_imgs.size(0)

        # Debug info
        # print(
        #     f"ER_SSA_GDM: buffer size={self.buffer_sample_count} "
        #     f"(min_score={self.buffer_scores.min():.3f}, max_score={self.buffer_scores.max():.3f})"
        # )

    def _sample_from_buffer(self, batch_size: int) -> Tensor:
        """バッファからランダムサンプリング (GDMではバッファ自体が分布近似されているため一様サンプリングでOK)"""
        if self.buffer_imgs is None or self.buffer_sample_count == 0:
            raise RuntimeError("Buffer is empty.")
        
        # 要求サイズがバッファより大きい場合は、重複を許してサンプリングするか、バッファサイズに制限するか
        # ここでは重複許容(replace=True)で要求サイズ分返す実装にする（元実装の挙動を維持）
        idxs = torch.randint(self.buffer_sample_count, size=(batch_size,))
        return self.buffer_imgs[idxs]


# import math
# from dataclasses import InitVar, dataclass, field
# from typing import Protocol, cast

# import torch
# from torch import Tensor, nn

# from methods.base import BaseTTA
# from .utils import get_pca_basis

# # --- Original Protocol & Classes ---

# class TensorCallable(Protocol):
#     def __call__(self, x: Tensor) -> Tensor: ...

# class SSALoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(
#         self,
#         mu_a: Tensor,
#         var_a: Tensor,
#         mu_b: Tensor,
#         var_b: Tensor,
#     ) -> Tensor:
#         var_a = var_a + 1e-6
#         var_b = var_b + 1e-6
#         diff = mu_a - mu_b
#         kl_ab = 0.5 * (torch.log(var_b / var_a)
#                        + (var_a + diff.pow(2)) / var_b
#                        - 1.0)
#         kl_ba = 0.5 * (torch.log(var_a / var_b)
#                        + (var_b + diff.pow(2)) / var_a
#                        - 1.0)
#         return (kl_ab + kl_ba).mean()

# # --- CMF Implementation ---

# @dataclass
# class CMFTracker:
#     """
#     Continual Momentum Filtering (CMF) Tracker for a single statistic vector.
#     Based on ICLR 2024 "Continual Momentum Filtering on Parameter Space..." Section 3.4.
#     """
#     num_features: int
#     alpha: float  # Hyperparam a: Source retention (0.0 to 1.0) [cite: 191]
#     q: float      # Hyperparam q: Adaptability/Noise variance [cite: 192]
#     device: torch.device
    
#     # Internal states
#     source_val: Tensor = field(init=False)      # phi^(0): Initial source statistic
#     mu_post: Tensor = field(init=False)         # mu_{t|t}: Posterior mean
#     sigma_post: Tensor = field(init=False)      # Sigma_{t|t}: Posterior covariance (scalar approx)
#     initialized: bool = field(init=False, default=False)

#     def __post_init__(self):
#         # [cite_start]r (observation noise) is constrained such that q + r = 1 [cite: 192]
#         self.r = 1.0 - self.q
#         self.sigma_post = torch.tensor(0.0, device=self.device) # Initial uncertainty is 0

#     def init_state(self, initial_value: Tensor) -> None:
#         """Initialize with source statistics (phi^0)."""
#         self.source_val = initial_value.clone().detach()
#         self.mu_post = initial_value.clone().detach()
#         self.sigma_post = torch.tensor(0.0, device=self.device)
#         self.initialized = True

#     def update(self, observation: Tensor) -> Tensor:
#         """
#         Apply Kalman Filter update step.
#         observation: theta^(t) (current batch statistic)
#         Returns: tilde_theta^(t) (filtered statistic)
#         """
#         if not self.initialized:
#             raise RuntimeError("CMFTracker must be initialized with source stats first.")

#         # [cite_start]--- Predict Step [cite: 179, 182] ---
#         # mu_{t|t-1} = alpha * mu_{t-1|t-1} + (1 - alpha) * phi^(0)
#         mu_pred = self.alpha * self.mu_post + (1 - self.alpha) * self.source_val
        
#         # Sigma_{t|t-1} = alpha^2 * Sigma_{t-1|t-1} + q
#         sigma_pred = (self.alpha ** 2) * self.sigma_post + self.q

#         # [cite_start]--- Update Step [cite: 179, 182] ---
#         # beta_t = r / (Sigma_{t|t-1} + r)
#         # Note: beta_t corresponds to (1 - KalmanGain)
#         beta_t = self.r / (sigma_pred + self.r)

#         # mu_{t|t} = beta_t * mu_{t|t-1} + (1 - beta_t) * theta^(t)
#         self.mu_post = beta_t * mu_pred + (1 - beta_t) * observation
        
#         # Sigma_{t|t} = beta_t * Sigma_{t|t-1}
#         self.sigma_post = beta_t * sigma_pred

#         # --- Parameter Ensemble (Optional in paper, implicit here) ---
#         # In this implementation, we treat the posterior mean (mu_{t|t}) 
#         # as the robust estimate of the global statistic.
#         return self.mu_post


# @dataclass
# class GlobalStatsCMF:
#     """
#     Replaces GlobalStats with CMF logic.
#     Tracks Mean and Variance independently using Kalman Filters.
#     """
#     num_features: int
#     cmf_alpha: float = 0.99   # Recommended alpha [cite: 241]
#     cmf_q: float = 0.005      # Recommended q [cite: 241]
#     device: torch.device = field(default_factory=lambda: torch.device("cpu"))

#     _mu_tracker: CMFTracker = field(init=False)
#     _var_tracker: CMFTracker = field(init=False)
    
#     mu: Tensor = field(init=False)   # Current global estimate
#     var: Tensor = field(init=False)  # Current global estimate
#     mu2: Tensor = field(init=False)  # Derived global mu2
#     initialized: bool = field(init=False, default=False)

#     def __post_init__(self) -> None:
#         self._mu_tracker = CMFTracker(self.num_features, self.cmf_alpha, self.cmf_q, self.device)
#         self._var_tracker = CMFTracker(self.num_features, self.cmf_alpha, self.cmf_q, self.device)
        
#         # Placeholders for access
#         self.mu = torch.zeros(self.num_features, device=self.device)
#         self.var = torch.ones(self.num_features, device=self.device)
#         self.mu2 = torch.ones(self.num_features, device=self.device)

#     def combine(self, batch_mu: Tensor, batch_var: Tensor) -> tuple[Tensor, Tensor, Tensor]:
#         """
#         Update global stats using CMF with the current batch observations.
#         Note: Unlike EMA, explicit alpha is not passed here as it's part of the CMF state.
#         """
#         if not self.initialized:
#             raise RuntimeError("GlobalStatsCMF must be initialized (commit) before combining.")

#         # Filter Mean
#         self.mu = self._mu_tracker.update(batch_mu)
        
#         # Filter Variance
#         # Note: We filter variance directly to ensure stability, 
#         # rather than filtering mu2 and deriving var (which can lead to negative var issues)
#         self.var = self._var_tracker.update(batch_var)
#         self.var = torch.clamp(self.var, min=1e-6)

#         # Derive mu2 for consistency with interface
#         self.mu2 = torch.clamp(self.var + self.mu.pow(2), min=1e-6)

#         return self.mu, self.mu2, self.var

#     @torch.no_grad()
#     def commit(self, mu: Tensor, var: Tensor) -> None:
#         """Initialize the 'Source' model statistics (phi^0)."""
#         if not self.initialized:
#             self._mu_tracker.init_state(mu)
#             self._var_tracker.init_state(var)
#             self.initialized = True
        
#         # Update current view
#         self.mu.copy_(mu.detach())
#         self.var.copy_(var.detach())
#         self.mu2.copy_(torch.clamp(var + mu.pow(2), min=1e-6))


# @dataclass
# class ER_SSA(BaseTTA):
#     """
#     ER_SSA with CMF (Continual Momentum Filtering) instead of EMA.
#     """
#     pc_config: InitVar[dict | None] = None

#     buffer_size: int = 512
#     min_buffer_size: int = 1
#     ssa_weight: float = 1.0
    
#     # CMF Hyperparameters
#     # [cite_start]alpha: 0.99 (Source retention), q: 0.005 (Adaptability) are defaults from paper [cite: 241]
#     cmf_alpha: float = 0.99
#     cmf_q: float = 0.005
    
#     main_loss: str = "none"

#     feature_extractor: TensorCallable | None = field(init=False, default=None, repr=False)
#     predictor: TensorCallable | None = field(init=False, default=None, repr=False)

#     feature_mean: Tensor | None = field(init=False, default=None, repr=False)
#     pca_basis: Tensor | None = field(init=False, default=None, repr=False)
#     pca_mean: Tensor | None = field(init=False, default=None, repr=False)
#     pca_var: Tensor | None = field(init=False, default=None, repr=False)
    
#     # Replaced GlobalStats with GlobalStatsCMF
#     global_stats: GlobalStatsCMF | None = field(init=False, default=None, repr=False)

#     ssa_loss_fn: SSALoss = field(init=False, repr=False)

#     buffer_imgs: Tensor | None = field(init=False, default=None, repr=False)
#     buffer_scores: Tensor | None = field(init=False, default=None, repr=False)
#     buffer_sample_count: int = field(init=False, default=0, repr=False)

#     def __post_init__(self, compile_model: dict | None, pc_config: dict | None):
#         self._pc_config = dict(pc_config or {})
#         super().__post_init__(compile_model)

#         if self.min_buffer_size < 1:
#             raise ValueError("min_buffer_size must be >= 1.")
#         if self.min_buffer_size > self.buffer_size:
#             raise ValueError("min_buffer_size cannot exceed buffer_size.")

#         self._init_subspace()

#         feature = getattr(self.net, "feature", None)
#         predictor = getattr(self.net, "predict_from_feature", None)
#         if not callable(feature) or not callable(predictor):
#             raise AttributeError("Net must define callable feature and predict_from_feature methods.")
#         self.feature_extractor = cast(TensorCallable, feature)
#         self.predictor = cast(TensorCallable, predictor)

#         self.ssa_loss_fn = SSALoss()

#     def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
#         if self.feature_extractor is None or self.predictor is None:
#             raise RuntimeError("ER_SSA feature extractor is not initialized.")

#         feats = self.feature_extractor(x)
#         flat_feats = feats.reshape(feats.size(0), -1)
#         y_pred = self.predictor(flat_feats).view(-1)

#         self._push_to_buffer(x)

#         if self.buffer_sample_count < self.min_buffer_size:
#             ssa_loss: Tensor = torch.tensor(0.0, device=x.device)
#             total_loss: Tensor | None = None
#             buffer_ready = False

#             # Even if buffer not ready, we maintain the stats tracking if possible,
#             # but usually we wait for buffer. Here we just strictly follow original logic flow.
#             # Initial commit is done in _init_subspace.
#         else:
#             x_buf = self._sample_from_buffer(x.shape[0]).to(self.device)
#             buf_mu, buf_var = self._compute_batch_stats(x_buf)

#             if self.pca_mean is None or self.pca_var is None:
#                 raise RuntimeError("PCA statistics are not initialized.")
#             if self.global_stats is None:
#                 raise RuntimeError("Global statistics are not initialized.")

#             # --- CMF Update ---
#             # Using CMF to combine current buffer stats with global history
#             global_mu, global_mu2, global_var = self.global_stats.combine(buf_mu, buf_var)
            
#             # Note: In CMF implementation, 'combine' updates the internal state automatically.
#             # We don't need an explicit 'commit' here unless we want to reset the source,
#             # which we don't. The CMF state evolves continuously.

#             ssa_loss = self.ssa_loss_fn(global_mu, global_var, self.pca_mean, self.pca_var)
#             total_loss = float(self.ssa_weight) * ssa_loss
#             buffer_ready = True

#         output = {
#             "y_pred": y_pred,
#             "ssa_loss": ssa_loss.detach() if 'ssa_loss' in locals() else torch.tensor(0., device=x.device),
#             "buffer_ready": torch.tensor(1.0 if buffer_ready else 0.0, device=x.device),
#         }
#         return output, total_loss

#     def _update(self, engine, batch):
#         if self.train_mode:
#             self.net.train()
#         else:
#             self.net.eval()

#         x, y = batch
#         x = x.to(self.device)
#         y = y.to(self.device).float().flatten()

#         if self.opt is not None:
#             self.opt.zero_grad(set_to_none=True)

#         output, loss = self.adapt_step(x, y)

#         if self.opt is not None and loss is not None:
#             loss.backward()
#             self.opt.step()

#         output["y"] = y
#         return output

#     def _init_subspace(self) -> None:
#         if not self._pc_config:
#             raise ValueError("pc_config must be provided.")
#         mean, basis, var = get_pca_basis(**self._pc_config)
#         self.feature_mean = mean.to(self.device)
#         self.pca_basis = basis.to(self.device)
#         self.pca_var = var.to(self.device)
#         self.pca_mean = torch.zeros_like(self.pca_var)
        
#         # Initialize GlobalStatsCMF with CMF params
#         stats = GlobalStatsCMF(
#             num_features=self.pca_basis.shape[1],
#             cmf_alpha=self.cmf_alpha,
#             cmf_q=self.cmf_q,
#             device=self.device,
#         )
#         # Commit initial source statistics (phi^0)
#         stats.commit(self.pca_mean, self.pca_var)
#         self.global_stats = stats

#     def _project_to_subspace(self, flat_feats: Tensor) -> Tensor:
#         if self.feature_mean is None or self.pca_basis is None:
#             raise RuntimeError("PCA subspace is not initialized.")
#         return (flat_feats - self.feature_mean) @ self.pca_basis

#     def _compute_batch_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
#         if self.feature_extractor is None:
#             raise RuntimeError("Feature extractor is not initialized.")
#         feats = self.feature_extractor(x)
#         flat_feats = feats.reshape(feats.size(0), -1)
#         feats_subspace = self._project_to_subspace(flat_feats)
#         mu = feats_subspace.mean(dim=0)
#         var = feats_subspace.var(dim=0, unbiased=True) + 1e-6
#         return mu, var

#     @torch.no_grad()
#     def _push_to_buffer(self, x: Tensor) -> None:
#         x_cpu = x.detach().cpu()
#         bsz = x_cpu.shape[0]
#         new_scores = torch.randn(bsz)

#         if self.buffer_imgs is None or self.buffer_scores is None:
#             self.buffer_imgs = x_cpu
#             self.buffer_scores = new_scores
#         else:
#             self.buffer_imgs = torch.cat([self.buffer_imgs, x_cpu], dim=0)
#             self.buffer_scores = torch.cat([self.buffer_scores, new_scores], dim=0)

#         current_size = self.buffer_scores.size(0)
#         if current_size > self.buffer_size:
#             _, top_indices = torch.topk(self.buffer_scores, k=self.buffer_size, sorted=False)
#             self.buffer_imgs = self.buffer_imgs[top_indices]
#             self.buffer_scores = self.buffer_scores[top_indices]
#         self.buffer_sample_count = self.buffer_imgs.size(0)

#     def _sample_from_buffer(self, batch_size: int) -> Tensor:
#         if self.buffer_imgs is None or self.buffer_sample_count == 0:
#             raise RuntimeError("Buffer is empty.")
#         idxs = torch.randint(self.buffer_sample_count, size=(batch_size,))
#         return self.buffer_imgs[idxs]
