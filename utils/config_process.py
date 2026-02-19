import json
import glob
import re
import yaml
from pathlib import Path

LAYOUT_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _dataset_info(config: dict) -> tuple[str, str | None]:
    """Extract (dataset_name, scene) from config."""
    dataset_name = config["dataset"]["name"]
    scene = (config["dataset"].get("config") or {}).get("scene")
    return dataset_name, scene


def _dataset_dir_name(dataset_cfg: dict) -> str:
    """Build dataset directory name from corruption settings.

    e.g. "utkface", "utkface-gaussian_noise-5"
    """
    dataset_name = dataset_cfg["name"]
    val_corruption = dataset_cfg.get("val_corruption") or {}
    severity = val_corruption.get("severity")
    corruption_name = (
        val_corruption.get("corruption_name") or val_corruption.get("name")
    )

    if corruption_name is not None and severity is not None:
        return f"{dataset_name}-{corruption_name}-{severity}"
    if severity is not None:
        return f"{dataset_name}-{severity}"
    return dataset_name


def resolve_train_dir(config: dict, base_dir: str) -> tuple[Path, str, str, Path]:
    """Resolve paths for source training.

    Returns:
        (run_dir, train_dir, val_dir, model_dir)
    """
    dataset_name, scene = _dataset_info(config)

    run_dir = Path(base_dir) / dataset_name
    train_dir = f"train_{dataset_name}"
    val_dir = f"val_{dataset_name}"
    model_dir = Path("models/weights", dataset_name)
    if scene is not None:
        run_dir = run_dir / scene
        model_dir = model_dir / scene

    return run_dir, train_dir, val_dir, model_dir


def resolve_tta_dir(
    config: dict,
    base_dir: str,
    stream_label: str,
    seed: int,
) -> Path:
    """Resolve output directory for TTA.

    Returns:
        output_dir
    """
    dataset_name, scene = _dataset_info(config)
    dataset_dir = _dataset_dir_name(config["dataset"])
    backbone = config["regressor"]["config"]["backbone"]

    tta_cfg = config.get("tta") or {}
    raw_method = tta_cfg.get("method")
    method_name = str(raw_method) if raw_method is not None else "base"
    seed_label = f"seed{seed}"

    if dataset_name == "utkface":
        return Path(base_dir, stream_label, dataset_dir,
                    f"{backbone}-{method_name}", seed_label)
    if dataset_name == "4seasons":
        seq_name = config["dataset"]["config"].get("seqs", ["unknown"])[0]
        return Path(base_dir, dataset_dir, scene or "unknown", seq_name,
                    f"{backbone}-{method_name}", seed_label)
    raise ValueError(f"Unsupported dataset: {dataset_name!r}")


def resolve_save_model_path(
    output_dir: Path,
    base_dir: str,
    backbone: str,
    method_key: str,
) -> Path:
    """Resolve the model weight save path for --save option.

    Converts output path structure to save path:
        results/<stream>/<dataset>/<method>/<seed>
        → models/tta_weights/<stream>/<dataset>/<backbone>_<method>.pt
    """

    rel_path = output_dir.relative_to(base_dir) # Get path relative to base_dir e.g. results/~ -> ~
    rel_parts = rel_path.parts # Split into parts: (stream, dataset, method, seed)
    save_subdir = Path(*rel_parts[:-2]) if len(rel_parts) >= 2 else Path() # Exclude last 2 parts (method, seed) to get save_subdir e.g. stream/dataset
    save_dir = Path("models", "tta_weights", save_subdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{backbone}_{method_key}.pt"


# ---------------------------------------------------------------------------
# File resolution (for feature / stats files)
# ---------------------------------------------------------------------------

def resolve_unique_file(root: Path, dataset: str, backbone: str) -> Path:
    """Find exactly one ``<backbone>.pt`` under ``<root>/<dataset>/``."""
    matches = sorted(glob.glob(str(root / dataset / "**" / f"{backbone}.pt"), recursive=True))
    assert len(matches) == 1, f"Expected 1 file, found {len(matches)} under {root}/{dataset}/{backbone}"
    return Path(matches[0])


def resolve_tta_features(
    root: Path, dataset: str, backbone: str, split: str, glob_pattern: str | None,
) -> list[Path]:
    """Glob TTA feature ``.pt`` files."""
    if glob_pattern:
        pattern = glob_pattern if Path(glob_pattern).is_absolute() else str(root / glob_pattern)
    else:
        pattern = str(root / "**" / f"{split}_features" / dataset / "**" / f"{backbone}.pt")
    return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]


def resolve_input_files(config: dict) -> tuple[Path, list[Path], Path]:
    """Resolve train feature file, TTA feature files, and stat file from config.

    Reads ``config["data"]`` section.
    """
    data_cfg = config.get("data") or {}
    dataset = data_cfg["dataset"]
    backbone = data_cfg["backbone"]
    train_root = Path(data_cfg.get("train_root", "models/train_features"))
    tta_root = Path(data_cfg.get("tta_root", "models/tta_features"))
    stat_root = Path(data_cfg.get("stat_root", "models/stats"))
    split = data_cfg.get("tta_split", "val")

    train_file = Path(data_cfg["train_file"]) if data_cfg.get("train_file") else resolve_unique_file(train_root, dataset, backbone)
    stat_file = Path(data_cfg["stat_file"]) if data_cfg.get("stat_file") else resolve_unique_file(stat_root, dataset, backbone)

    if data_cfg.get("tta_files"):
        tta_files = [Path(p) for p in data_cfg["tta_files"]]
    else:
        tta_files = resolve_tta_features(tta_root, dataset, backbone, split, data_cfg.get("tta_glob"))
    assert tta_files, "No TTA feature files found."

    return train_file, tta_files, stat_file

def resolve_grid(plot_cfg: dict, nplots: int) -> tuple[int, int, Path]:
    """Resolve grid layout and output path from plot config.

    Returns:
        (nrows, ncols, output_path)

    If layout is specified (e.g. "2x2"), it is also appended to the output filename.
        e.g. output="outputs/umap.png", layout="2x2" → outputs/umap_2x2.png
    """
    layout = plot_cfg.get("layout")
    output = Path(plot_cfg.get("output", "outputs/umap_features.png"))

    if layout is None:
        return 1, nplots, output

    m = LAYOUT_RE.match(layout)
    assert m, f"layout must be ROWSxCOLS, got {layout!r}"
    nrows, ncols = int(m.group(1)), int(m.group(2))
    assert nrows * ncols == nplots, f"layout {nrows}x{ncols} != {nplots} plots"

    suffix = layout.strip().lower().replace(" ", "")
    output = output.with_name(f"{output.stem}_{suffix}{output.suffix or '.png'}")

    return nrows, ncols, output


# ---------------------------------------------------------------------------
# Data stream
# ---------------------------------------------------------------------------

def build_stream_label(data_stream_cfg: dict, seed: int) -> tuple[str, dict | None]:
    """Build stream label string and non-iid kwargs from data_stream config.

    Returns:
        (stream_label, non_iid_kwargs or None)
    """
    stream_type = (data_stream_cfg.get("type") or "iid").lower()

    if stream_type == "iid":
        return stream_type, None

    if stream_type != "non_iid":
        raise ValueError(
            f"Unsupported data stream type: {stream_type!r}. "
            "Use 'iid' or 'non_iid'."
        )

    cycles = data_stream_cfg.get("cycles") or data_stream_cfg.get("period", 1)
    sigma = (
        data_stream_cfg.get("sigma")
        or data_stream_cfg.get("sigma_label")
        or data_stream_cfg.get("beta", 1.0)
    )

    non_iid_kwargs = {
        "mode": data_stream_cfg.get("mode", "linear"),
        "sigma": sigma,
        "cycles": cycles,
        "length": data_stream_cfg.get("length"),
        "seed": data_stream_cfg.get("seed", seed),
    }

    stream_parts = ["non_iid", non_iid_kwargs["mode"]]
    if sigma is not None:
        stream_parts.append(f"sigma{sigma}")
    if cycles not in (None, 1):
        stream_parts.append(f"cycles{cycles}")

    return "-".join(map(str, stream_parts)), non_iid_kwargs


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_config(config: dict, output_dir: Path) -> None:
    """Save config.yaml to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Save metrics dict as JSON to output directory."""
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)