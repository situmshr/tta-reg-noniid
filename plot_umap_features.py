#!/usr/bin/env python3
"""Plot 2D UMAP embeddings with PCA projection (subplots per method)."""

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import umap
except ImportError as exc:  # pragma: no cover - optional dependency
    umap = None
    _UMAP_IMPORT_ERROR = exc
else:
    _UMAP_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
except ImportError as exc:  # pragma: no cover - optional dependency
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None


DEFAULT_LABEL_DIMS = {
    "utkface": 1,
    "4seasons": 6,
}
FOURSEASONS_SEQS = ("neighborhood_4", "neighborhood_6", "neighborhood_7")
FOURSEASONS_METHODS = ("base", "ssa", "er_ssa")
SSA_OFFLINE_LABEL = "ssa(offline)"
KNOWN_METHODS = (
    "er_ssa",
    "mem_ssa",
    "ada_ssa",
    "wssa",
    "adabn",
    "vm",
    "ssa",
    "base",
)
METHOD_ORDER = (
    "base",
    "ssa",
    "er_ssa",
    SSA_OFFLINE_LABEL,
    "wssa",
    "ada_ssa",
    "adabn",
    "vm",
    "mem_ssa",
    "unknown",
)
DISPLAY_METHOD_LABELS = {
    "base": "Source",
    "ssa": "SSA",
    "er_ssa": "RS-SSA",
    SSA_OFFLINE_LABEL: "SSA(stationary)",
}
NEIGHBORHOOD_RE = re.compile(r"^neighborhood_\d+$")
LAYOUT_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D UMAP embeddings from train_features and tta_features (subplots per method)."
    )
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g., utkface, 4seasons).")
    parser.add_argument("--backbone", default=None, help="Backbone name (e.g., resnet50-GN).")
    parser.add_argument("--train-root", type=Path, default=Path("models/train_features"))
    parser.add_argument("--tta-root", type=Path, default=Path("models/tta_features"))
    parser.add_argument("--stat-root", type=Path, default=Path("models/stats"))
    parser.add_argument("--train-file", type=Path, default=None, help="Explicit train feature file.")
    parser.add_argument("--tta-files", type=Path, nargs="*", default=None, help="Explicit TTA feature files.")
    parser.add_argument("--tta-glob", type=str, default=None, help="Glob pattern under tta-root.")
    parser.add_argument("--stat-file", type=Path, default=None, help="Explicit PCA stat file (mean/basis/eigvals).")
    parser.add_argument("--tta-split", default="val", help="Split name in tta_features (default: val).")
    parser.add_argument("--label-dim", type=int, default=None, help="Label dimension appended to features.")
    parser.add_argument("--feature-dim", type=int, default=None, help="Feature dimension (overrides label-dim).")
    parser.add_argument("--pca-dim", type=int, default=100, help="Top-K PCA dimensions for projection.")
    parser.add_argument("--max-samples", type=int, default=3000, help="Max samples per group.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", default="euclidean", help="UMAP metric.")
    parser.add_argument("--point-size", type=float, default=6.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Scatter alpha.")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--no-figure-title", action="store_true", help="Disable figure title.")
    parser.add_argument("--corruption", default=None, help="Filter to a single corruption.")
    parser.add_argument("--plot-methods", nargs="*", default=None, help="Methods to plot (order).")
    parser.add_argument("--layout", default=None, help="Grid layout as ROWSxCOLS (e.g., 2x2).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/umap_features.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def resolve_single_feature(root: Path, dataset: str, backbone: str) -> Path:
    pattern = str(root / dataset / "**" / f"{backbone}.pt")
    matches = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
    if not matches:
        raise FileNotFoundError(f"No feature file found under {root} for {dataset}/{backbone}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple feature files found for {dataset}/{backbone}: "
            + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def resolve_stat_file(root: Path, dataset: str, backbone: str) -> Path:
    pattern = str(root / dataset / "**" / f"{backbone}.pt")
    matches = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
    if not matches:
        raise FileNotFoundError(f"No stat file found under {root} for {dataset}/{backbone}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple stat files found for {dataset}/{backbone}: "
            + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def resolve_tta_features(root: Path, dataset: str, backbone: str, split: str, glob_pattern: str | None) -> List[Path]:
    if glob_pattern:
        pattern = glob_pattern if Path(glob_pattern).is_absolute() else str(root / glob_pattern)
        return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
    pattern = str(root / "**" / f"{split}_features" / dataset / "**" / f"{backbone}.pt")
    return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]


def split_features(
    data: torch.Tensor,
    feature_dim: int | None,
    label_dim: int | None,
) -> torch.Tensor:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(data.shape)}")
    if feature_dim is not None:
        if data.shape[1] <= feature_dim:
            raise ValueError(f"feature-dim={feature_dim} >= total dims {data.shape[1]}")
        return data[:, :feature_dim]
    if label_dim is None:
        raise ValueError("label-dim is required when feature-dim is not provided.")
    if data.shape[1] <= label_dim:
        raise ValueError(f"label-dim={label_dim} >= total dims {data.shape[1]}")
    return data[:, :-label_dim]


def load_feature_file(path: Path, feature_dim: int | None, label_dim: int | None) -> np.ndarray:
    data = torch.load(path, map_location="cpu").float()
    features = split_features(data, feature_dim, label_dim)
    return features.numpy()


def subsample(features: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or features.shape[0] <= max_samples:
        return features
    rng = np.random.default_rng(seed)
    idx = rng.choice(features.shape[0], size=max_samples, replace=False)
    return features[idx]


def infer_method(path: Path) -> str:
    # Prefer the longest known method (e.g., er_ssa over ssa) based on path parts.
    for part in path.parts:
        for method in KNOWN_METHODS:
            if part == method:
                return method
            if part.endswith(f"_{method}") or part.endswith(f"-{method}"):
                return method
    return "unknown"


def infer_sequence(path: Path) -> str | None:
    for part in path.parts:
        if NEIGHBORHOOD_RE.match(part):
            return part
    return None


def infer_corruption(path: Path, dataset: str | None) -> str | None:
    if dataset:
        prefix = f"{dataset}-"
        for part in path.parts:
            if not part.startswith(prefix):
                continue
            remainder = part[len(prefix):]
            corruption, sep, severity = remainder.rpartition("-")
            if sep and severity.isdigit():
                return corruption
    for part in path.parts:
        stem, sep, severity = part.rpartition("-")
        if sep and severity.isdigit() and "-" in stem:
            return stem.split("-", 1)[1]
    return None


def infer_stream_type(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("non_iid"):
            return "non_iid"
        if part == "iid" or part.startswith("iid-"):
            return "iid"
    return None


def normalize_method_label(method: str, stream_type: str | None) -> str:
    if method == "ssa" and stream_type == "iid":
        return SSA_OFFLINE_LABEL
    return method


def group_tta_by_method(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for p in paths:
        method = infer_method(p)
        stream_type = infer_stream_type(p)
        label = normalize_method_label(method, stream_type)
        grouped.setdefault(label, []).append(p)
    return grouped


def group_tta_by_corruption_and_method(
    paths: Sequence[Path],
    dataset: str | None,
) -> Dict[Tuple[str, str], List[Path]]:
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for p in paths:
        corruption = infer_corruption(p, dataset) or "unknown"
        method = infer_method(p)
        stream_type = infer_stream_type(p)
        label = normalize_method_label(method, stream_type)
        grouped.setdefault((corruption, label), []).append(p)
    return grouped


def group_tta_by_seq_and_method(
    paths: Sequence[Path],
    seqs: Sequence[str],
    methods: Sequence[str],
) -> Dict[Tuple[str, str], List[Path]]:
    grouped: Dict[Tuple[str, str], List[Path]] = {(s, m): [] for s in seqs for m in methods}
    for p in paths:
        seq = infer_sequence(p)
        method = infer_method(p)
        if seq is None or seq not in seqs or method not in methods:
            continue
        grouped[(seq, method)].append(p)
    return grouped


def sort_methods(methods: Sequence[str]) -> List[str]:
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda m: (order.get(m, len(order)), m))


def resolve_label_dim(dataset: str | None, feature_dim: int | None, label_dim: int | None) -> int | None:
    if feature_dim is not None:
        return label_dim
    resolved = label_dim if label_dim is not None else DEFAULT_LABEL_DIMS.get(dataset or "")
    if resolved is None:
        raise SystemExit("Provide --label-dim or --feature-dim.")
    if resolved <= 0:
        raise SystemExit("--label-dim must be a positive integer.")
    return resolved


def load_group_features(
    paths: Sequence[Path],
    feature_dim: int | None,
    label_dim: int | None,
    max_samples: int,
    seed: int,
) -> np.ndarray | None:
    if not paths:
        return None
    chunks = []
    for idx, path in enumerate(paths):
        feats = load_feature_file(path, feature_dim, label_dim)
        feats = subsample(feats, max_samples, seed + idx + 1)
        chunks.append(feats)
    combined = np.concatenate(chunks, axis=0)
    return subsample(combined, max_samples, seed + 101)


def load_pca_stats(stat_file: Path, pca_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    # Load precomputed PCA stats (mean, basis, eigvals) from models/stats.
    if pca_dim <= 0:
        raise SystemExit("--pca-dim must be a positive integer.")
    payload = torch.load(stat_file, map_location="cpu")
    mean = payload["mean"].float()
    basis = payload["basis"].float()
    eigvals = payload.get("eigvals")
    if eigvals is not None:
        eigvals = eigvals.float()
        k = min(pca_dim, eigvals.numel())
        top_idx = torch.topk(eigvals, k=k).indices
        basis = basis[:, top_idx]
    else:
        k = min(pca_dim, basis.shape[1])
        basis = basis[:, -k:]
    return mean.numpy(), basis.numpy()


def project_pca(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (features - mean) @ basis


def fit_shared_umap(source_pca: np.ndarray, args: argparse.Namespace) -> Tuple["umap.UMAP", np.ndarray]:
    # Learn a single UMAP mapping shared across all TTA methods.
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    source_emb = reducer.fit_transform(source_pca)
    return reducer, source_emb


def embed_target(
    reducer: "umap.UMAP",
    target_features: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    target_pca = project_pca(target_features, mean, basis)
    return reducer.transform(target_pca)


def compute_axis_limits(embeddings: Sequence[np.ndarray], pad: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_pts = np.concatenate(embeddings, axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def sanitize_filename_component(name: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", name).strip("_")
    return sanitized or "unknown"


def format_method_label(method: str) -> str:
    label = DISPLAY_METHOD_LABELS.get(method)
    if label:
        return label
    return method.upper()


def parse_layout(layout: str | None, nplots: int) -> Tuple[int, int] | None:
    if layout is None:
        return None
    match = LAYOUT_RE.match(layout)
    if not match:
        raise SystemExit("--layout must be in ROWSxCOLS format (e.g., 2x2).")
    nrows, ncols = int(match.group(1)), int(match.group(2))
    if nrows <= 0 or ncols <= 0:
        raise SystemExit("--layout values must be positive.")
    if nrows * ncols != nplots:
        raise SystemExit(f"--layout expects {nplots} slots, got {nrows * ncols}.")
    return nrows, ncols


def resolve_plot_methods(methods: Sequence[str] | None) -> List[str]:
    if not methods:
        return ["base", "ssa", "er_ssa", SSA_OFFLINE_LABEL]
    return [m.strip() for m in methods if m.strip()]


def plot_source_target(
    ax: "plt.Axes",
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    args: argparse.Namespace,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    # Plot precomputed embeddings in the shared UMAP space.
    ax.scatter(source_emb[:, 0], source_emb[:, 1], s=args.point_size, alpha=args.alpha, label="source")
    ax.scatter(target_emb[:, 0], target_emb[:, 1], s=args.point_size, alpha=args.alpha, label="target")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title, fontsize=28)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not args.no_legend:
        ax.legend(fontsize=16, markerscale=2.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def plot_source_targets(
    ax: "plt.Axes",
    source_emb: np.ndarray,
    target_embs: Dict[str, np.ndarray],
    args: argparse.Namespace,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    method_order: Sequence[str],
    method_colors: Dict[str, Tuple[float, float, float, float]],
) -> None:
    ax.scatter(
        source_emb[:, 0],
        source_emb[:, 1],
        s=args.point_size,
        alpha=args.alpha,
        label="source",
        color="black",
    )
    for method in method_order:
        target_emb = target_embs.get(method)
        if target_emb is None:
            continue
        ax.scatter(
            target_emb[:, 0],
            target_emb[:, 1],
            s=args.point_size,
            alpha=args.alpha,
            label=format_method_label(method),
            color=method_colors.get(method),
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not args.no_legend:
        ax.legend(fontsize=8, markerscale=1.2)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)


def ensure_deps() -> None:
    if umap is None:
        raise SystemExit(f"umap-learn is required: {_UMAP_IMPORT_ERROR}")
    if plt is None:
        raise SystemExit(f"matplotlib is required: {_PLOT_IMPORT_ERROR}")


def main() -> None:
    args = parse_args()
    ensure_deps()

    dataset = args.dataset
    backbone = args.backbone

    if args.train_file is None:
        if not dataset or not backbone:
            raise SystemExit("Provide --train-file or (--dataset and --backbone).")
        train_file = resolve_single_feature(args.train_root, dataset, backbone)
    else:
        train_file = args.train_file

    if args.tta_files is None or len(args.tta_files) == 0:
        if not dataset or not backbone:
            raise SystemExit("Provide --tta-files or (--dataset and --backbone).")
        tta_files = resolve_tta_features(args.tta_root, dataset, backbone, args.tta_split, args.tta_glob)
    else:
        tta_files = list(args.tta_files)

    if not tta_files:
        raise SystemExit("No TTA feature files found.")

    label_dim = resolve_label_dim(dataset, args.feature_dim, args.label_dim)

    source_features = load_feature_file(train_file, args.feature_dim, label_dim)
    source_features = subsample(source_features, args.max_samples, args.seed)

    if args.stat_file is None:
        if not dataset or not backbone:
            raise SystemExit("Provide --stat-file or (--dataset and --backbone).")
        stat_file = resolve_stat_file(args.stat_root, dataset, backbone)
    else:
        stat_file = args.stat_file

    # 1) PCA projection using precomputed stats, then learn a shared UMAP mapping.
    pca_mean, pca_basis = load_pca_stats(stat_file, args.pca_dim)
    source_pca = project_pca(source_features, pca_mean, pca_basis)
    reducer, source_emb = fit_shared_umap(source_pca, args)

    print(f"Loaded {len(tta_files)} TTA feature files.")

    if dataset == "4seasons":
        # For 4Seasons neighborhood, plot 3x3 grid: neighborhoods (rows) x methods (cols).
        grouped = group_tta_by_seq_and_method(tta_files, FOURSEASONS_SEQS, FOURSEASONS_METHODS)
        nrows, ncols = len(FOURSEASONS_SEQS), len(FOURSEASONS_METHODS)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

        target_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
        for seq in FOURSEASONS_SEQS:
            for method in FOURSEASONS_METHODS:
                paths = grouped.get((seq, method), [])
                print(f"Loading features for {seq} / {method}: {len(paths)} files")
                feats = load_group_features(paths, args.feature_dim, label_dim, args.max_samples, args.seed)
                if feats is None:
                    continue
                target_embeddings[(seq, method)] = embed_target(reducer, feats, pca_mean, pca_basis)

        if not target_embeddings:
            raise SystemExit("No target features loaded from TTA files.")
        xlim, ylim = compute_axis_limits([source_emb, *target_embeddings.values()])

        for r, seq in enumerate(FOURSEASONS_SEQS):
            for c, method in enumerate(FOURSEASONS_METHODS):
                ax = axes[r][c]
                target_emb = target_embeddings.get((seq, method))
                if target_emb is None:
                    ax.set_title(f"{seq} / {format_method_label(method)} (no data)")
                    ax.axis("off")
                    continue
                plot_source_target(
                    ax,
                    source_emb,
                    target_emb,
                    args,
                    title=f"{seq} / {format_method_label(method)}",
                    xlim=xlim,
                    ylim=ylim,
                )
    else:
        # Default: save one image per corruption (subplots per method).
        grouped = group_tta_by_corruption_and_method(tta_files, dataset)
        if args.corruption:
            grouped = {k: v for k, v in grouped.items() if k[0] == args.corruption}
            if not grouped:
                raise SystemExit(f"No TTA files found for corruption '{args.corruption}'.")
        target_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
        print(f"Found {len(grouped)} corruption/method groups.")
        total_groups = len(grouped)
        for idx, ((corruption, method), paths) in enumerate(grouped.items(), start=1):
            print(f"[{idx}/{total_groups}] Loading {corruption} / {method}: {len(paths)} files")
            feats = load_group_features(paths, args.feature_dim, label_dim, args.max_samples, args.seed)
            if feats is None:
                continue
            target_embeddings[(corruption, method)] = embed_target(reducer, feats, pca_mean, pca_basis)
        if not target_embeddings:
            raise SystemExit("No target features loaded from TTA files.")

        corruptions = sorted({c for c, _ in target_embeddings.keys()})
        methods_present = sort_methods({m for _, m in target_embeddings.keys()})  # type: ignore
        plot_methods = resolve_plot_methods(args.plot_methods)
        if missing_methods := [m for m in plot_methods if m not in methods_present]:
            print(f"Warning: missing methods (will show as empty subplots): {', '.join(missing_methods)}")

        xlim, ylim = compute_axis_limits([source_emb, *target_embeddings.values()])
        print(f"Computed axis limits: xlim={xlim}, ylim={ylim}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        multi = len(corruptions) > 1

        for corruption in corruptions:
            layout = parse_layout(args.layout, len(plot_methods))
            if layout is None:
                nrows, ncols = 1, len(plot_methods)
            else:
                nrows, ncols = layout
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(6 * ncols, 5 * nrows),
                squeeze=False,
            )

            for idx, method in enumerate(plot_methods):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                target_emb = target_embeddings.get((corruption, method))
                if target_emb is None:
                    ax.set_title(f"{format_method_label(method)} (no data)")
                    ax.axis("off")
                    continue
                plot_source_target(
                    ax,
                    source_emb,
                    target_emb,
                    args,
                    title=format_method_label(method),
                    xlim=xlim,
                    ylim=ylim,
                )

            if not args.no_figure_title:
                figure_title = f"{args.title} / {corruption}" if args.title else corruption
                fig.suptitle(figure_title)

            safe_corruption = sanitize_filename_component(corruption)
            if multi:
                suffix = args.output.suffix or ".png"
                out_path = args.output.with_name(f"{args.output.stem}_{safe_corruption}{suffix}")
            else:
                out_path = args.output

            rect = (0, 0, 1, 0.96) if not args.no_figure_title else (0, 0, 1, 1)
            fig.tight_layout(rect=rect)
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved UMAP plot to {out_path}")
        return

    if args.title and not args.no_figure_title:
        fig.suptitle(args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rect = (0, 0, 1, 0.96) if args.title and not args.no_figure_title else (0, 0, 1, 1)
    fig.tight_layout(rect=rect)
    fig.savefig(args.output, dpi=200)
    print(f"Saved UMAP plot to {args.output}")


if __name__ == "__main__":
    main()
