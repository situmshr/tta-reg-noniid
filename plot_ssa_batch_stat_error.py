#!/usr/bin/env python3
"""Plot error between SSA subspace minibatch stats and source stats.

This script loads pretrained weights, projects minibatch features into the
SSA (PCA) subspace, computes minibatch mean/variance, and measures error to
source stats (mean=0, var=pc_vars) using either L2 or symmetric KL. It does not
update any model parameters.

Example:
  python plot_ssa_batch_stat_error.py \
    --config-iid configs/tta/utkface/utkface-iid-Res50-GN-ssa-Lv5.yaml \
    --config-non-iid configs/tta/utkface/utkface-non_iid-Res50-GN-ssa-Lv5.yaml \
    --output outputs/plots/utkface_gn_ssa_batch_stat_error.png \
    --output-time outputs/plots/utkface_gn_ssa_batch_stat_error_time.png
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

from data import get_datasets, get_non_iid_dataset
from methods.utils import diagonal_gaussian_kl_loss, get_pca_basis
from models.arch import create_regressor
from utils.seed import fix_seed

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Run with the repo venv, e.g. `./.venv/bin/python ...`."
    ) from exc


DEFAULT_IID_CFG = "configs/tta/utkface/utkface-iid-Res50-GN-ssa-Lv5.yaml"
DEFAULT_NONIID_CFG = "configs/tta/utkface/utkface-non_iid-Res50-GN-ssa-Lv5.yaml"
METRIC_LABELS = {
    "l2": "L2",
    "sym_kl": "Sym KL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot minibatch SSA subspace stats error vs. source stats (UTKFace GN)."
    )
    parser.add_argument(
        "--config-iid",
        type=Path,
        default=Path(DEFAULT_IID_CFG),
        help=f"IID config file (default: {DEFAULT_IID_CFG}).",
    )
    parser.add_argument(
        "--config-non-iid",
        type=Path,
        default=Path(DEFAULT_NONIID_CFG),
        help=f"Non-IID config file (default: {DEFAULT_NONIID_CFG}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs", "plots", "utkface_gn_ssa_batch_stat_error.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--output-time",
        type=Path,
        default=None,
        help="Output path for the time-series plot (default: derive from --output).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=tuple(METRIC_LABELS.keys()),
        default="l2",
        help="Distance metric for minibatch stats (default: l2).",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bin count.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of minibatches to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for non-iid stream when not specified in the config.",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"Device for inference (default: {default_device}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory for the DataLoader.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def resolve_non_iid_kwargs(data_stream_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    cycles = data_stream_cfg.get("cycles")
    if cycles is None:
        cycles = data_stream_cfg.get("period", 1)

    sigma = data_stream_cfg.get("sigma")
    if sigma is None:
        sigma = data_stream_cfg.get("sigma_label")
    if sigma is None:
        sigma = data_stream_cfg.get("beta", 1.0)

    return {
        "mode": data_stream_cfg.get("mode", "linear"),
        "sigma": sigma,
        "cycles": cycles,
        "length": data_stream_cfg.get("length"),
        "seed": data_stream_cfg.get("seed", seed),
    }


def build_val_loader(
    config: dict[str, Any],
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    _, val_ds = get_datasets(config)

    data_stream_cfg = config["dataset"].get("data_stream") or {}
    stream_type = (data_stream_cfg.get("type") or "iid").lower()
    if stream_type == "non_iid":
        non_iid_kwargs = resolve_non_iid_kwargs(data_stream_cfg, seed)
        print(f"Applying non-iid data stream: {non_iid_kwargs}")
        val_ds = get_non_iid_dataset(val_ds, **non_iid_kwargs)
    elif stream_type != "iid":
        raise ValueError(f"Unsupported data stream type: {stream_type!r}. Use 'iid' or 'non_iid'.")

    dl_cfg = dict(config.get("adapt_dataloader") or {})
    dl_cfg.setdefault("batch_size", 64)
    dl_cfg.setdefault("shuffle", False)
    dl_cfg.setdefault("num_workers", num_workers)
    dl_cfg.setdefault("pin_memory", pin_memory)
    return DataLoader(val_ds, **dl_cfg)


def load_pca_stats(config: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    tta_cfg = config.get("tta") or {}
    pc_cfg = tta_cfg.get("pc_config")
    if not pc_cfg:
        raise ValueError("tta.pc_config is required to define the SSA subspace.")
    mean, basis, var = get_pca_basis(**pc_cfg)
    return mean.to(device), basis.to(device), var.to(device)


def load_regressor(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    regressor = create_regressor(config).to(device)
    state = torch.load(config["regressor"]["source"], map_location="cpu")
    regressor.load_state_dict(state)
    regressor.eval()
    for param in regressor.parameters():
        param.requires_grad_(False)
    return regressor


def symmetric_kl(
    m1: Tensor,
    v1: Tensor,
    m2: Tensor,
    v2: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    v1 = v1.clamp(min=eps)
    v2 = v2.clamp(min=eps)
    forward = diagonal_gaussian_kl_loss(m1, v1, m2, v2, eps=eps, dim_reduction="sum")
    backward = diagonal_gaussian_kl_loss(m2, v2, m1, v1, eps=eps, dim_reduction="sum")
    return forward + backward


def compute_batch_errors(
    metric: str,
    f_pc_mean: Tensor,
    f_pc_var: Tensor,
    src_var: Tensor,
) -> tuple[Tensor, Tensor]:
    if metric == "l2":
        mean_err = torch.linalg.norm(f_pc_mean, ord=2)
        var_err = torch.linalg.norm(f_pc_var - src_var, ord=2)
        return mean_err, var_err
    if metric == "sym_kl":
        zeros = torch.zeros_like(f_pc_mean)
        # Isolate mean shift by using the source variance for both distributions.
        mean_err = symmetric_kl(f_pc_mean, src_var, zeros, src_var)
        var_err = symmetric_kl(zeros, f_pc_var, zeros, src_var)
        return mean_err, var_err
    raise ValueError(f"Unsupported metric: {metric!r}.")


@torch.no_grad()
def collect_batch_errors(
    regressor: torch.nn.Module,
    dataloader: DataLoader,
    mean: Tensor,
    basis: Tensor,
    var: Tensor,
    device: torch.device,
    max_batches: int | None,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_errors: list[float] = []
    var_errors: list[float] = []
    weights: list[int] = []

    feature_fn = getattr(regressor, "feature", None)
    if not callable(feature_fn):
        raise AttributeError("Regressor must define a callable feature(x) method.")

    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        feat = feature_fn(x)
        f_pc = (feat - mean) @ basis # type: ignore
        f_pc_mean = f_pc.mean(dim=0)
        f_pc_var = f_pc.var(dim=0, unbiased=False)

        mean_err, var_err = compute_batch_errors(metric, f_pc_mean, f_pc_var, var)
        mean_errors.append(float(mean_err))
        var_errors.append(float(var_err))
        weights.append(int(x.shape[0]))

    return (
        np.asarray(mean_errors, dtype=np.float64),
        np.asarray(var_errors, dtype=np.float64),
        np.asarray(weights, dtype=np.int64),
    )


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    if values.size == 0:
        return np.asarray([np.nan for _ in qs], dtype=np.float64)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    out = []
    for q in qs:
        idx = np.searchsorted(cum, q * total)
        idx = min(idx, v.size - 1)
        out.append(v[idx])
    return np.asarray(out, dtype=np.float64)


def summarize_errors(errors: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float, int]:
    total_samples = int(weights.sum()) if weights.size else 0
    mean = float(np.average(errors, weights=weights)) if weights.size else float("nan")
    q10, q50, q90 = weighted_quantiles(errors, weights, [0.1, 0.5, 0.9])
    return mean, q10, q50, q90, total_samples


def print_summary(
    name: str,
    mean_errors: np.ndarray,
    var_errors: np.ndarray,
    weights: np.ndarray,
    metric_label: str,
) -> None:
    mean_mean, mean_q10, mean_q50, mean_q90, total_samples = summarize_errors(mean_errors, weights)
    var_mean, var_q10, var_q50, var_q90, _ = summarize_errors(var_errors, weights)
    print(
        f"{name}: metric={metric_label}, batches={mean_errors.size}, samples={total_samples}, "
        f"mean_err(mean)={mean_mean:.4f}, q10={mean_q10:.4f}, median={mean_q50:.4f}, q90={mean_q90:.4f}"
    )
    print(
        f"{name}: metric={metric_label}, batches={var_errors.size}, samples={total_samples}, "
        f"mean_err(var)={var_mean:.4f}, q10={var_q10:.4f}, median={var_q50:.4f}, q90={var_q90:.4f}"
    )


def plot_histograms(
    state: tuple[np.ndarray, np.ndarray, np.ndarray],
    non_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    bins: int,
    output: Path,
    metric_label: str,
) -> None:
    state_mean, state_var, state_weights = state
    non_mean, non_var, non_weights = non_state
    if state_mean.size == 0 or non_mean.size == 0:
        raise ValueError("No errors collected; check dataloader and dataset.")

    mean_edges = np.histogram_bin_edges(np.concatenate([state_mean, non_mean]), bins=bins)
    var_edges = np.histogram_bin_edges(np.concatenate([state_var, non_var]), bins=bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(
        state_mean,
        bins=mean_edges,
        weights=state_weights,
        color="#4C72B0",
        alpha=0.6,
        label="stationary",
    )
    axes[0].hist(
        non_mean,
        bins=mean_edges,
        weights=non_weights,
        color="#DD8452",
        alpha=0.6,
        label="non-stationary",
    )
    axes[0].set_title("mean")
    axes[0].set_xlabel(f"{metric_label} error")
    axes[0].set_ylabel("sample count")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].hist(
        state_var,
        bins=var_edges,
        weights=state_weights,
        color="#4C72B0",
        alpha=0.6,
        label="stationary",
    )
    axes[1].hist(
        non_var,
        bins=var_edges,
        weights=non_weights,
        color="#DD8452",
        alpha=0.6,
        label="non-stationary",
    )
    axes[1].set_title("variance")
    axes[1].set_xlabel(f"{metric_label} error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("SSA subspace minibatch stats vs. source (UTKFace GN)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_time_series(
    state: tuple[np.ndarray, np.ndarray, np.ndarray],
    non_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    metric_label: str,
) -> None:
    state_mean, state_var, state_weights = state
    non_mean, non_var, non_weights = non_state
    if state_mean.size == 0 or non_mean.size == 0:
        raise ValueError("No errors collected; check dataloader and dataset.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_state = np.cumsum(state_weights) - state_weights
    t_non = np.cumsum(non_weights) - non_weights

    axes[0].plot(t_state, state_mean, color="#4C72B0", linewidth=1.5, label="stationary")
    axes[0].plot(t_non, non_mean, color="#DD8452", linewidth=1.5, label="non-stationary")
    axes[0].set_title("mean")
    axes[0].set_ylabel(f"{metric_label} error")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t_state, state_var, color="#4C72B0", linewidth=1.5, label="stationary")
    axes[1].plot(t_non, non_var, color="#DD8452", linewidth=1.5, label="non-stationary")
    axes[1].set_title("variance")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(f"{metric_label} error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("SSA subspace minibatch stats vs. source (UTKFace GN)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def derive_time_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}_time{output.suffix}")
    return output.with_name(f"{output.name}_time.png")


def run_for_config(
    config_path: Path,
    seed: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    max_batches: int | None,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = load_config(config_path)
    dl = build_val_loader(config, seed, num_workers, pin_memory)
    mean, basis, var = load_pca_stats(config, device)
    regressor = load_regressor(config, device)
    return collect_batch_errors(regressor, dl, mean, basis, var, device, max_batches, metric)


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)

    device = torch.device(args.device)
    metric_label = METRIC_LABELS[args.metric]
    iid_errors = run_for_config(
        args.config_iid,
        args.seed,
        device,
        args.num_workers,
        args.pin_memory,
        args.max_batches,
        args.metric,
    )
    non_errors = run_for_config(
        args.config_non_iid,
        args.seed,
        device,
        args.num_workers,
        args.pin_memory,
        args.max_batches,
        args.metric,
    )

    print_summary("state", *iid_errors, metric_label)
    print_summary("non-state", *non_errors, metric_label)
    plot_histograms(iid_errors, non_errors, args.bins, args.output, metric_label)
    print(f"saved plot: {args.output}")
    time_output = args.output_time or derive_time_output_path(args.output)
    plot_time_series(iid_errors, non_errors, time_output, metric_label)
    print(f"saved time-series plot: {time_output}")


if __name__ == "__main__":
    main()
