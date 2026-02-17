import json
import yaml
from pathlib import Path


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