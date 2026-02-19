import argparse
import re
from pprint import pprint
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt

from methods.utils.pca_basis import get_pca_basis
from utils.config_process import load_config, resolve_input_files

plt.switch_backend("Agg")

DEFAULT_LABEL_DIMS = {"utkface": 1, "4seasons": 6}
FOURSEASONS_SEQS = ("neighborhood_4", "neighborhood_6", "neighborhood_7")
FOURSEASONS_METHODS = ("base", "ssa", "rs_ssa")
SSA_OFFLINE_LABEL = "ssa(offline)"
DEFAULT_PLOT_METHODS = ("base", "ssa", "rs_ssa", SSA_OFFLINE_LABEL)
KNOWN_METHODS = ("rs_ssa", "ssa", "base")
DISPLAY_METHOD_LABELS = {
    "base": "Source", "ssa": "SSA", "rs_ssa": "RS-SSA",
    SSA_OFFLINE_LABEL: "SSA(stationary)",
}
NEIGHBORHOOD_RE = re.compile(r"^neighborhood_\d+$")
LAYOUT_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 2D UMAP embeddings.")
    p.add_argument("-c", required=True, help="Config file path.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature processing
# ---------------------------------------------------------------------------

def split_features(data: torch.Tensor, feature_dim: int | None, label_dim: int | None) -> torch.Tensor:
    assert data.ndim == 2
    if feature_dim is not None:
        return data[:, :feature_dim]
    assert label_dim is not None and label_dim > 0
    return data[:, :-label_dim]


def load_feature_file(path: Path, feature_dim: int | None, label_dim: int | None) -> np.ndarray:
    data = torch.load(path, map_location="cpu").float()
    return split_features(data, feature_dim, label_dim).numpy()


def subsample(features: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or features.shape[0] <= max_samples:
        return features
    idx = np.random.default_rng(seed).choice(features.shape[0], size=max_samples, replace=False)
    return features[idx]


# ---------------------------------------------------------------------------
# Path inference
# ---------------------------------------------------------------------------

def infer_method(path: Path) -> str:
    for part in path.parts:
        for method in KNOWN_METHODS:
            if part == method or part.endswith(f"_{method}") or part.endswith(f"-{method}"):
                return method
    return "unknown"


def infer_sequence(path: Path) -> str | None:
    return next((p for p in path.parts if NEIGHBORHOOD_RE.match(p)), None)


def infer_corruption(path: Path, dataset: str | None) -> str | None:
    if dataset:
        prefix = f"{dataset}-"
        for part in path.parts:
            if not part.startswith(prefix):
                continue
            corruption, sep, severity = part[len(prefix):].rpartition("-")
            if sep and severity.isdigit():
                return corruption
    return None


def infer_stream_type(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("non_iid"):
            return "non_iid"
        if part == "iid" or part.startswith("iid-"):
            return "iid"
    return None


def normalize_method_label(method: str, stream_type: str | None) -> str:
    return SSA_OFFLINE_LABEL if method == "ssa" and stream_type == "iid" else method


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_corruption_and_method(paths: Sequence[Path], dataset: str | None) -> Dict[Tuple[str, str], List[Path]]:
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for p in paths:
        corruption = infer_corruption(p, dataset) or "unknown"
        label = normalize_method_label(infer_method(p), infer_stream_type(p))
        grouped.setdefault((corruption, label), []).append(p)
    return grouped


def group_by_seq_and_method(paths: Sequence[Path], seqs: Sequence[str], methods: Sequence[str]) -> Dict[Tuple[str, str], List[Path]]:
    grouped: Dict[Tuple[str, str], List[Path]] = {(s, m): [] for s in seqs for m in methods}
    for p in paths:
        seq, method = infer_sequence(p), infer_method(p)
        if seq in seqs and method in methods:
            grouped[(seq, method)].append(p)  # type: ignore
    return grouped


# ---------------------------------------------------------------------------
# PCA / UMAP
# ---------------------------------------------------------------------------

def load_pca_stats(stat_file: Path, pca_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    mean, basis, _ = get_pca_basis(str(stat_file), contrib_top_k=pca_dim)
    return mean.numpy(), basis.numpy()


def project_pca(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (features - mean) @ basis


def fit_shared_umap(source_pca: np.ndarray, config: dict) -> Tuple[umap.UMAP, np.ndarray]:
    umap_cfg = config.get("umap") or {}
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        metric=umap_cfg.get("metric", "euclidean"),
        random_state=config.get("seed", 7),
    )
    return reducer, np.asarray(reducer.fit_transform(source_pca))


def embed_target(reducer: umap.UMAP, features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(reducer.transform(project_pca(features, mean, basis)))


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def compute_axis_limits(embeddings: Sequence[np.ndarray], pad: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = np.concatenate(embeddings, axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    p = (mx - mn) * pad
    return (mn[0] - p[0], mx[0] + p[0]), (mn[1] - p[1], mx[1] + p[1])


def format_method_label(method: str) -> str:
    return DISPLAY_METHOD_LABELS.get(method, method.upper())


def resolve_grid(layout: str | None, nplots: int) -> Tuple[int, int]:
    if layout is None:
        return 1, nplots
    m = LAYOUT_RE.match(layout) # e.g. "2x3" → m.group(1)="2", m.group(2)="3"
    assert m, f"layout must be ROWSxCOLS, got {layout!r}"
    nrows, ncols = int(m.group(1)), int(m.group(2))
    assert nrows * ncols == nplots, f"layout {nrows}x{ncols} != {nplots} plots"
    return nrows, ncols


def plot_source_target(ax, source_emb, target_emb, plot_cfg: dict, title, xlim, ylim):
    point_size = plot_cfg.get("point_size", 6.0)
    alpha = plot_cfg.get("alpha", 0.6)
    no_legend = plot_cfg.get("no_legend", False)

    ax.scatter(source_emb[:, 0], source_emb[:, 1], s=point_size, alpha=alpha, label="source")
    ax.scatter(target_emb[:, 0], target_emb[:, 1], s=point_size, alpha=alpha, label="target")
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title=title, xlim=xlim, ylim=ylim)
    if not no_legend:
        ax.legend(fontsize=16, markerscale=2.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def save_figure(fig, output: Path, title: str | None, no_figure_title: bool):
    output.parent.mkdir(parents=True, exist_ok=True)
    if title and not no_figure_title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title and not no_figure_title else 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved UMAP plot to {output}")


# ---------------------------------------------------------------------------
# Source embedding
# ---------------------------------------------------------------------------

def build_source_embedding(config: dict, train_file: Path, stat_file: Path):
    feat_cfg = config.get("feature") or {}
    feature_dim = feat_cfg.get("feature_dim")
    label_dim = feat_cfg.get("label_dim") or DEFAULT_LABEL_DIMS.get((config.get("data") or {}).get("dataset", ""))
    pca_dim = feat_cfg.get("pca_dim", 100)
    max_samples = feat_cfg.get("max_samples", 10000)
    seed = config.get("seed", 7)

    source = subsample(load_feature_file(train_file, feature_dim, label_dim), max_samples, seed)
    pca_mean, pca_basis = load_pca_stats(stat_file, pca_dim)
    reducer, source_emb = fit_shared_umap(project_pca(source, pca_mean, pca_basis), config)
    return feature_dim, label_dim, pca_mean, pca_basis, reducer, source_emb


def load_embedded_group(
    paths: Sequence[Path], config: dict,
    feature_dim: int | None, label_dim: int | None,
    reducer, pca_mean: np.ndarray, pca_basis: np.ndarray,
) -> np.ndarray | None:
    if not paths:
        return None
    feat_cfg = config.get("feature") or {}
    max_samples = feat_cfg.get("max_samples", 10000)
    seed = config.get("seed", 7)

    chunks = [subsample(load_feature_file(p, feature_dim, label_dim), max_samples, seed + i + 1) for i, p in enumerate(paths)]
    combined = subsample(np.concatenate(chunks), max_samples, seed + 101)
    return embed_target(reducer, combined, pca_mean, pca_basis)


# ---------------------------------------------------------------------------
# Grid plots
# ---------------------------------------------------------------------------

def plot_4seasons_grid(config: dict, tta_files, feature_dim, label_dim, reducer, pca_mean, pca_basis, source_emb):
    plot_cfg = config.get("plot") or {}
    grouped = group_by_seq_and_method(tta_files, FOURSEASONS_SEQS, FOURSEASONS_METHODS)

    embeddings = {}
    for (seq, method), paths in grouped.items():
        emb = load_embedded_group(paths, config, feature_dim, label_dim, reducer, pca_mean, pca_basis)
        if emb is not None:
            embeddings[(seq, method)] = emb

    xlim, ylim = compute_axis_limits([source_emb, *embeddings.values()])
    nrows, ncols = len(FOURSEASONS_SEQS), len(FOURSEASONS_METHODS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for row, seq in enumerate(FOURSEASONS_SEQS):
        for col, method in enumerate(FOURSEASONS_METHODS):
            ax = axes[row][col]
            emb = embeddings.get((seq, method))
            if emb is None:
                ax.set_title(f"{seq} / {format_method_label(method)} (no data)")
                ax.axis("off")
            else:
                plot_source_target(ax, source_emb, emb, plot_cfg,
                                   f"{seq} / {format_method_label(method)}", xlim, ylim)

    output = Path(plot_cfg.get("output", "outputs/umap_features.png"))
    save_figure(fig, output, plot_cfg.get("title"), plot_cfg.get("no_figure_title", False))


def plot_corruption_grids(config: dict, tta_files, feature_dim, label_dim, reducer, pca_mean, pca_basis, source_emb):
    data_cfg = config.get("data") or {}
    plot_cfg = config.get("plot") or {}
    dataset = data_cfg.get("dataset")
    output = Path(plot_cfg.get("output", "outputs/umap_features.png"))

    grouped = group_by_corruption_and_method(tta_files, dataset)
    corruption_filter = plot_cfg.get("corruption")
    if corruption_filter:
        grouped = {k: v for k, v in grouped.items() if k[0] == corruption_filter}

    embeddings = {}
    for (corruption, method), paths in grouped.items():
        emb = load_embedded_group(paths, config, feature_dim, label_dim, reducer, pca_mean, pca_basis)
        if emb is not None:
            embeddings[(corruption, method)] = emb

    plot_methods = plot_cfg.get("plot_methods") or list(DEFAULT_PLOT_METHODS)
    plot_methods = [m.strip() for m in plot_methods if m.strip()]
    xlim, ylim = compute_axis_limits([source_emb, *embeddings.values()])
    nrows, ncols = resolve_grid(plot_cfg.get("layout"), len(plot_methods))
    title = plot_cfg.get("title")
    no_figure_title = plot_cfg.get("no_figure_title", False)

    for corruption in sorted({c for c, _ in embeddings}):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        for idx, method in enumerate(plot_methods):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            emb = embeddings.get((corruption, method))
            if emb is None:
                ax.set_title(f"{format_method_label(method)} (no data)")
                ax.axis("off")
            else:
                plot_source_target(ax, source_emb, emb, plot_cfg, format_method_label(method), xlim, ylim)

        figure_title = f"{title} / {corruption}" if title else corruption
        if len(set(c for c, _ in embeddings)) > 1:
            safe = re.sub(r"[^\w.-]+", "_", corruption).strip("_") or "unknown"
            out_path = output.with_name(f"{output.stem}_{safe}{output.suffix or '.png'}")
        else:
            out_path = output
        save_figure(fig, out_path, figure_title, no_figure_title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.c)

    train_file, tta_files, stat_file = resolve_input_files(config)
    feature_dim, label_dim, pca_mean, pca_basis, reducer, source_emb = build_source_embedding(config, train_file, stat_file)

    dataset = (config.get("data") or {}).get("dataset")

    if dataset == "utkface":
        plot_corruption_grids(config, tta_files, feature_dim, label_dim, reducer, pca_mean, pca_basis, source_emb)
    elif dataset == "4seasons":
        plot_4seasons_grid(config, tta_files, feature_dim, label_dim, reducer, pca_mean, pca_basis, source_emb)

if __name__ == "__main__":
    main()
