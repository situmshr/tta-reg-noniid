import argparse
import json
import yaml
import torch
import wandb
from pathlib import Path
from torch import Tensor

class FixedBLR:
    """
    BLR with optional fixed weights.
    Calculates posterior precision (L) for uncertainty, and can optionally learn mean (mu).
    """
    def __init__(self, sigma2=1.0, jitter=1e-6):
        self.sigma2 = float(sigma2)
        self.jitter = float(jitter)
        self.L: Tensor | None = None
        self.mu: Tensor | None = None

    def _augment(self, Phi: Tensor, add_bias: bool) -> Tensor:
        if not add_bias:
            return Phi
        ones = torch.ones(Phi.size(0), 1, device=Phi.device, dtype=Phi.dtype)
        return torch.cat([Phi, ones], dim=1)

    def fit_precision(self,
                      Phi: Tensor,
                      fixed_w: Tensor,
                      fixed_b: Tensor | None = None,
                      y: Tensor | None = None,
                      learn_mu: bool = False,
                      add_bias: bool | None = None,
                      alpha: float = 0.0,
                      alpha_bias: float = 0.0):
        """
        Phi: (N, D) Projected features
        fixed_w: (K, D) Projected weights
        fixed_b: (K,) or (1, K) Bias
        """
        device = Phi.device
        if add_bias is None:
            add_bias = fixed_b is not None

        Phi = self._augment(Phi, add_bias)
        _, D = Phi.shape

        # 精度行列 A の Cholesky 分解 (予測分散の計算用)
        # A = 1/sigma2 * Phi.T @ Phi + jitter * I
        PhiTPhi = Phi.T @ Phi
        A = PhiTPhi / self.sigma2
        if alpha > 0.0 or (add_bias and alpha_bias > 0.0):
            alpha_vec = torch.full((D,), float(alpha), device=device, dtype=Phi.dtype)
            if add_bias:
                alpha_vec[-1] = float(alpha_bias)
            A = A + torch.diag(alpha_vec)
        A = A + self.jitter * torch.eye(D, device=device, dtype=Phi.dtype)
        self.L = torch.linalg.cholesky(A)

        if learn_mu:
            if y is None:
                raise ValueError("y must be provided when learn_mu=True.")
            y = y.float()
            if y.ndim == 1:
                y = y[:, None]
            b = (Phi.T @ y) / self.sigma2
            self.mu = torch.cholesky_solve(b, self.L)
        else:
            if add_bias:
                if fixed_b is None:
                    raise ValueError("fixed_b must be provided when add_bias=True and learn_mu=False.")
                fixed_b = fixed_b.view(-1, 1) if fixed_b.ndim == 1 else fixed_b.T
                self.mu = torch.cat([fixed_w.T, fixed_b], dim=0)
            else:
                self.mu = fixed_w.T

    @torch.no_grad()
    def predict(self, Phi_test: Tensor):
        device = Phi_test.device
        # バイアス項の拡張
        if self.mu.shape[0] > Phi_test.shape[1]: # bias exists
            Phi_test = torch.cat(
                [Phi_test, torch.ones(Phi_test.shape[0], 1, device=device, dtype=Phi_test.dtype)],
                dim=1,
            )

        # Mean: y = Phi @ mu
        mean = Phi_test @ self.mu # (M, K)

        # Variance: sigma^2 + Phi @ A^-1 @ Phi.T
        # Efficient comp: v = sum((Phi @ L^-T)^2, dim=1)
        Ainv_PhiT = torch.cholesky_solve(Phi_test.T, self.L)
        quad = (Phi_test * Ainv_PhiT.T).sum(dim=1)
        std = torch.sqrt((quad + self.sigma2).clamp_min(1e-9))

        return mean.squeeze(), std

def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)

def load_features(config, split):
    dataset = config["dataset"]["name"]
    backbone = config["regressor"]["config"]["backbone"]
    subdir = "train_features" if split == "train" else "valid_features"
    path = Path("models", subdir, dataset, f"{backbone}.pt")
    
    data = torch.load(path, map_location="cpu").float()
    dim = int(config["regressor"]["config"]["feature_dim"])
    return data[:, :dim], data[:, dim:]

def get_pca_basis(feats: Tensor, k: int):
    mean = feats.mean(dim=0, keepdim=True)
    centered = feats - mean
    _, _, V = torch.pca_lowrank(centered, q=min(k, feats.shape[1]))
    return mean, V.float()

def load_pretrained_weights(config):
    # Locate checkpoint
    dataset = config["dataset"]["name"]
    backbone = config["regressor"]["config"]["backbone"]
    search_dir = Path("models", "weights", dataset)
    if "gender" in config.get("dataset", {}).get("config", {}):
        search_dir /= config["dataset"]["config"]["gender"]
    
    candidates = list(search_dir.glob(f"{backbone}_model_*.pt"))
    candidates = [p for p in candidates if "-vbll" not in p.stem]
    if not candidates: raise FileNotFoundError("Base checkpoint not found")
    
    # Simple logic to pick latest
    ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading weights from: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    return sd["regressor.weight"], sd.get("regressor.bias")

@torch.no_grad()
def estimate_sigma2_from_predictions(preds: Tensor, targets: Tensor) -> float:
    targets = targets.float()
    if preds.ndim == 1 and targets.ndim > 1:
        preds = preds.view_as(targets)
    elif preds.ndim > 1 and targets.ndim == 1:
        targets = targets.view_as(preds)
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {targets.shape}")
    resid = preds - targets
    return float(resid.pow(2).mean().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True)
    parser.add_argument("-o", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.c)
    torch.manual_seed(args.seed)
    
    # 1. Load Data & Weights
    train_x, train_y = load_features(config, "train")
    val_x, val_y = load_features(config, "val")
    base_w, base_b = load_pretrained_weights(config)

    # 2. PCA & Projection
    pc_cfg = (config.get("blr") or {}).get("pc_config") or config.get("tta", {}).get("pc_config")
    k = int(pc_cfg.get("contrib_top_k", 16))
    
    pc_mean, pc_basis = get_pca_basis(train_x, k)
    
    # Project Data: Z = X @ V
    # Note: Original code used `feats @ pc_basis` (no centering) for projection
    train_z = train_x @ pc_basis
    val_z = val_x @ pc_basis

    # Project Weights: W_sub = W_orig @ V
    # (K, D) @ (D, k) -> (K, k)
    fixed_w_sub = base_w @ pc_basis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blr_cfg = config.get("blr") or {}
    learn_mu = bool(blr_cfg.get("learn_mu", False))
    add_bias = blr_cfg.get("add_bias")
    if add_bias is None:
        add_bias = base_b is not None
    alpha = float(blr_cfg.get("alpha", 0.0))
    alpha_bias = float(blr_cfg.get("alpha_bias", 0.0))
    estimate_sigma2 = blr_cfg.get("estimate_sigma2", True)
    sigma2 = blr_cfg.get("sigma2", 1.0)
    train_z_device = train_z.to(device)
    train_y_device = train_y.to(device)
    fixed_w_device = fixed_w_sub.to(device)
    base_b_device = base_b.to(device) if base_b is not None else None

    # 4. Setup BLR (Fixed Weights)
    out_dir = Path(args.o, config["dataset"]["name"], f"{config['regressor']['config']['backbone']}-blr-fixed")
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="blr-fixed-eval", config=config, dir=out_dir)

    blr = FixedBLR(sigma2=sigma2)
    blr.fit_precision(
        train_z_device,
        fixed_w_device,
        base_b_device,
        y=train_y_device,
        learn_mu=learn_mu,
        add_bias=add_bias,
        alpha=alpha,
        alpha_bias=alpha_bias,
    )
    if estimate_sigma2:
        pred_train, _ = blr.predict(train_z_device)
        sigma2 = estimate_sigma2_from_predictions(pred_train, train_y_device)
        config.setdefault("blr", {})["sigma2"] = sigma2
        print(f"Estimated sigma2 from training residuals: {sigma2:.6f}")
        if sigma2 != blr.sigma2:
            blr.sigma2 = sigma2
            blr.fit_precision(
                train_z_device,
                fixed_w_device,
                base_b_device,
                y=train_y_device,
                learn_mu=learn_mu,
                add_bias=add_bias,
                alpha=alpha,
                alpha_bias=alpha_bias,
            )
    
    # 5. Predict & Evaluate
    pred, std = blr.predict(val_z.to(device))
    pred = pred.cpu()
    std = std.cpu()
    
    diff = pred - val_y.squeeze()
    metrics = {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt(diff.pow(2).mean()).item(),
        "mean_std": std.mean().item(),
        "sigma2": float(sigma2),
    }
    print(json.dumps(metrics, indent=2))

    # 5. Save
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    ckpt_dir = Path("models", "weights", config["dataset"]["name"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config['regressor']['config']['backbone']}_blr_model.pt"
    torch.save({
        "mu": blr.mu,
        "L": blr.L,
        "pc_mean": pc_mean,
        "pc_basis": pc_basis,
        "config": {"sigma2": float(sigma2)},
    }, ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    main()
