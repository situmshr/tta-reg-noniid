from dataclasses import dataclass, field, InitVar
from typing import Protocol, cast
import copy

import torch
from torch import Tensor, nn

from methods.base import BaseTTA
from .utils import get_pca_basis, diagonal_gaussian_kl_loss


@torch.no_grad()
def ema_update_model(model_to_update: nn.Module,
                     model_to_merge: nn.Module,
                     momentum: float,
                     device: torch.device,
                     update_all: bool = False) -> nn.Module:
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(),
                                                   model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + \
                    (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update


class TensorCallable(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class KLCallable(Protocol):
    def __call__(self, m1: Tensor, v1: Tensor, m2: Tensor, v2: Tensor) -> Tensor: ...


class GlobalStats:
    """Count-weighted statistics tracker (mean / meansq / count)."""

    def __init__(self, num_features: int, device: torch.device) -> None:
        self.device = device
        self.num_features = num_features
        self.reset()

    def reset(self) -> None:
        self.mean = torch.zeros(self.num_features, device=self.device)
        self.meansq = torch.zeros(self.num_features, device=self.device)
        self.count = torch.tensor(0.0, device=self.device)  # effective samples

    @torch.no_grad()
    def init_from(self, mean: Tensor, var: Tensor, count: float = 1.0) -> None:
        count_t = torch.tensor(float(count), device=self.device)
        self.mean = mean.to(self.device)
        self.meansq = (var.to(self.device) + mean.to(self.device).pow(2))
        self.count = count_t

    @torch.no_grad()
    def add_batch(self, feats: Tensor) -> None:
        feats = feats.to(self.device)
        bsz = float(feats.shape[0])
        if bsz <= 0:
            return
        batch_mean = feats.mean(dim=0)
        batch_meansq = feats.pow(2).mean(dim=0)
        total = self.count + bsz
        self.mean = self.count / total * self.mean + bsz / total * batch_mean
        self.meansq = self.count / total * self.meansq + bsz / total * batch_meansq
        self.count = total

    @torch.no_grad()
    def remove_batch(self, feats: Tensor) -> None:
        feats = feats.to(self.device)
        bsz = float(feats.shape[0])
        if bsz <= 0 or self.count.item() <= 0:
            return
        batch_mean = feats.mean(dim=0)
        batch_meansq = feats.pow(2).mean(dim=0)
        remaining = self.count - bsz
        if remaining < 1e-6:
            self.reset()
            self.count = torch.tensor(1e-6, device=self.device)
            return
        self.mean = (self.count * self.mean - bsz * batch_mean) / remaining
        self.meansq = (self.count * self.meansq - bsz * batch_meansq) / remaining
        self.count = remaining

    @torch.no_grad()
    def compute(self) -> tuple[Tensor, Tensor]:
        count = torch.clamp(self.count, min=1e-6)
        var = torch.clamp(self.meansq - self.mean.pow(2), min=1e-6)
        return self.mean, var


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
    Experience Replay SSA with count-weighted global stats and feature-aware buffer.
    """

    pc_config: InitVar[dict | None] = None
    loss_config: InitVar[dict | None] = None

    buffer_size: int = 512
    min_buffer_size: int = 1
    ssa_weight: float = 1.0
    ema_alpha: float = 0.1     # retained for API compatibility (not used)
    ema_momentum: float = 0.99  # retained for API compatibility (not used)
    main_loss: str = "none"    # retained for API compatibility (not used)
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

    buffer_imgs: Tensor | None = field(init=False, default=None, repr=False)
    buffer_feats: Tensor | None = field(init=False, default=None, repr=False)  # PCA-space features
    buffer_age: Tensor | None = field(init=False, default=None, repr=False)
    buffer_sample_count: int = field(init=False, default=0, repr=False)
    source_net: nn.Module | None = field(init=False, default=None, repr=False)

    def __post_init__(self, compile_model: dict | None, val_dataset, target_names, pc_config: dict | None, loss_config: dict | None):
        self._pc_config = dict(pc_config or {})
        self._loss_config = dict(loss_config or {})
        super().__post_init__(compile_model, val_dataset, target_names)

        if self.min_buffer_size < 1:
            raise ValueError("min_buffer_size must be >= 1.")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError("min_buffer_size cannot exceed buffer_size.")

        self._init_subspace()
        self.source_net = copy.deepcopy(self.net).to(self.device)
        for p in self.source_net.parameters():  # type: ignore
            p.requires_grad_(False)

        feature = getattr(self.net, "feature", None)
        predictor = getattr(self.net, "predict_from_feature", None)
        if not callable(feature) or not callable(predictor):
            raise AttributeError("Net must define callable feature and predict_from_feature methods.")
        self.feature_extractor = cast(TensorCallable, feature)
        self.predictor = cast(TensorCallable, predictor)

    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        if self.feature_extractor is None or self.predictor is None:
            raise RuntimeError("ER_SSA feature extractor is not initialized.")
        if self.global_stats is None or self.loss_fn is None or self.dim_weight is None:
            raise RuntimeError("ER_SSA is not initialized.")
        
        # --- 1. 特徴量抽出とバッファサンプリングの準備 ---
        # ここではデータの準備だけを行い、GlobalStatsやバッファの更新はまだ行いません
        
        # 後で更新処理に使う変数を初期化
        current_batch_feats_sub = None
        current_buf_feats_sub = None
        buf_idx = None
        
        if self.buffer_sample_count < self.min_buffer_size:
            feats = self.feature_extractor(x)
            flat_feats = feats.reshape(feats.size(0), -1)
            feats_subspace = self._project_to_subspace(flat_feats)
            y_pred = self.predictor(flat_feats).view(-1)
            buffer_ready = False
            
            # 更新用データ
            current_batch_feats_sub = feats_subspace
            
        else:
            buf_imgs, buf_idx = self._sample_from_buffer(x.shape[0])
            concat_x = torch.cat([x, buf_imgs.to(self.device)], dim=0)
            feats = self.feature_extractor(concat_x)
            flat_feats = feats.reshape(feats.size(0), -1)
            feats_subspace = self._project_to_subspace(flat_feats)

            # バッチ部分とバッファ部分に分割
            current_batch_feats_sub = feats_subspace[: x.shape[0]]
            current_buf_feats_sub = feats_subspace[x.shape[0]:]

            # 古いバッファの特徴量の破棄
            if self.global_stats is not None and buf_idx is not None and current_buf_feats_sub is not None:
                self.global_stats.remove_batch(self.buffer_feats[buf_idx].to(self.device))  # type: ignore

            # 予測は現在のバッチ分のみ
            y_pred = self.predictor(flat_feats).view(-1)[: x.shape[0]]
            buffer_ready = True

        # --- 2. SSA Loss の計算 (更新前の GlobalStats を使用) ---
        
        # 重要: ここで更新前の状態を取得（スナップショット）
        global_mu, global_var = self.global_stats.compute()

        # combined_mu の計算
        c1 = float(self.global_stats.count.item())
        c2 = float(feats_subspace.shape[0])
        total = max(c1 + c2, 1e-6)
        
        batch_mean = feats_subspace.mean(dim=0)
        batch_var = feats_subspace.var(dim=0, unbiased=True) + 1e-6
        
        # 更新前の global_mu を使って結合統計量を計算
        combined_mu = (c1 / total) * global_mu + (c2 / total) * batch_mean
        combined_mu2 = (c1 / total) * (global_var + global_mu.pow(2)) + (c2 / total) * (batch_var + batch_mean.pow(2))
        combined_var = combined_mu2 - combined_mu.pow(2)

        kl_forward = self.loss_fn(combined_mu, combined_var, self.pca_mean, self.pca_var) # type: ignore
        kl_backward = self.loss_fn(self.pca_mean, self.pca_var, combined_mu, combined_var) # type: ignore
        ssa_loss = (kl_forward + kl_backward) * self.dim_weight
        ssa_loss = ssa_loss.sum()
        total_loss = float(self.ssa_weight) * ssa_loss

        # --- 3. Global Stats とバッファの更新 (Loss計算後) ---

        if self.buffer_sample_count < self.min_buffer_size:
            # バッファが足りない場合は現在のバッチを追加するのみ
            self._push_to_buffer(x, current_batch_feats_sub)
        else:
            # バッファが十分ある場合の更新処理（ご質問の箇所）
            if self.global_stats is not None and buf_idx is not None and current_buf_feats_sub is not None:
                # 2. リフレッシュされた特徴量を統計に追加
                self.global_stats.add_batch(current_buf_feats_sub.detach())
            
            # バッファ配列自体の更新
            # self.buffer_imgs[buf_idx] = ... (画像は不変なので省略可)
            if self.buffer_feats is not None and buf_idx is not None:
                self.buffer_feats[buf_idx] = current_buf_feats_sub.detach().cpu() # type: ignore
                self.buffer_age[buf_idx] = 0 # type: ignore

            # 3. 現在のバッチをバッファに追加（これ内部でも global_stats.add_batch が呼ばれる）
            self._push_to_buffer(x, current_batch_feats_sub)

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
        y = y.to(self.device).float()

        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

        output, loss = self.adapt_step(x, y)

        if self.opt is not None and loss is not None:
            # print(f"ER_SSA_GDM loss: {loss.item():.6f}")
            # 元の更新のみ
            # loss.backward()
            # self.opt.step()
            loss.backward()
            self.opt.step()
            if self.source_net is not None:
                ema_update_model(
                    model_to_update=self.net,
                    model_to_merge=self.source_net,
                    momentum=self.ema_momentum,
                    device=self.device,
                    update_all=False,
                )

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

        stats = GlobalStats(num_features=self.pca_basis.shape[1], device=self.device)
        stats.init_from(self.pca_mean, self.pca_var, count=1e-6)
        self.global_stats = stats

        with torch.no_grad():
            regressor_weight = self._get_regressor_weight()
            dim_weight = torch.abs(regressor_weight @ self.pca_basis).sum(dim=0)
            self.dim_weight = (dim_weight + self.weight_bias).pow(self.weight_exp)
            print(f"SSA dim_weight: {self.dim_weight.shape}")

    def _project_to_subspace(self, flat_feats: Tensor) -> Tensor:
        if self.feature_mean is None or self.pca_basis is None:
            raise RuntimeError("PCA subspace is not initialized.")
        return (flat_feats - self.feature_mean) @ self.pca_basis

    def _get_regressor_weight(self) -> Tensor:
        regressor_module = getattr(self.net, "regressor", None)
        if regressor_module is None or not isinstance(regressor_module, nn.Module):
            raise AttributeError("Net must expose a regressor module with weight parameter.")
        weight = getattr(regressor_module, "weight", None)
        if not isinstance(weight, Tensor):
            raise AttributeError("Regressor module is missing weight tensor.")
        return weight

    @torch.no_grad()
    def _push_to_buffer(self, x: Tensor, feats_subspace: Tensor) -> None:
        """
        Reservoir-style buffer update with scores; keep (x, encoder output) pairs
        and maintain global stats. When over capacity, drop lowest-score samples.
        """
        x_cpu = x.detach().cpu()
        feat_cpu = feats_subspace.detach().cpu()
        bsz = x_cpu.shape[0]

        # Increment age for existing entries
        if self.buffer_age is not None:
            self.buffer_age += 1

        # Random scores for new samples (higher is better)
        new_scores = torch.randn(bsz)

        if self.buffer_imgs is None:
            self.buffer_imgs = x_cpu
            self.buffer_feats = feat_cpu
            self.buffer_age = torch.zeros(bsz, dtype=torch.long)
            self.buffer_scores = new_scores
        else:
            self.buffer_imgs = torch.cat([self.buffer_imgs, x_cpu], dim=0)
            self.buffer_feats = torch.cat([self.buffer_feats, feat_cpu], dim=0) # type: ignore
            self.buffer_age = torch.cat(
                [self.buffer_age, torch.zeros(bsz, dtype=torch.long)], dim=0 # type: ignore
            ) # type: ignore
            self.buffer_scores = torch.cat([self.buffer_scores, new_scores], dim=0)  # type: ignore[attr-defined]

        # Add new feats to global stats
        if self.global_stats is not None:
            self.global_stats.add_batch(feat_cpu.to(self.device))

        # Capacity control: keep highest-score samples
        current_size = self.buffer_scores.size(0)  # type: ignore[union-attr]
        if current_size > self.buffer_size:
            _, top_idx = torch.topk(self.buffer_scores, k=self.buffer_size, sorted=False)  # type: ignore[union-attr]
            keep_mask = torch.zeros(current_size, dtype=torch.bool)
            keep_mask[top_idx] = True

            # Remove stats for discarded samples
            if self.global_stats is not None:
                drop_feats = self.buffer_feats[~keep_mask].to(self.device) # type: ignore
                if drop_feats.numel() > 0:
                    self.global_stats.remove_batch(drop_feats)

            self.buffer_imgs = self.buffer_imgs[keep_mask]
            self.buffer_feats = self.buffer_feats[keep_mask] # type: ignore
            self.buffer_age = self.buffer_age[keep_mask] # type: ignore
            self.buffer_scores = self.buffer_scores[keep_mask]  # type: ignore[union-attr]

        self.buffer_sample_count = int(self.buffer_feats.shape[0]) if self.buffer_feats is not None else 0

    def _sample_from_buffer(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Sample imgs with bias toward older samples (age-proportional), return (imgs, idx)."""
        if self.buffer_feats is None or self.buffer_sample_count == 0:
            raise RuntimeError("Buffer is empty.")
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor is not initialized.")
        weights = (self.buffer_age.float() + 1).clamp(min=1e-3) # type: ignore
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, num_samples=min(batch_size, self.buffer_sample_count), replacement=False)
        # Return sampled images; caller handles feature refresh and age reset.
        return self.buffer_imgs[idx], idx  # type: ignore