import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from densratio import densratio
from torch import Tensor
import yaml

from data import get_datasets
from utils.seed import fix_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute density-ratio-weighted feature stats for Biwi Kinect sequences."
    )
    p.add_argument("-c", "--config", required=True, help="Config file (YAML/JSON)")
    p.add_argument("-o", "--output", required=True, help="Output root directory")
    p.add_argument("--target-person", required=True, action="append", default=[],
                   help="Person ID to define target sequence (can be repeated).")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def weighted_mean_cov(features: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
    weights = weights.to(features.dtype)
    w_sum = weights.sum()
    mean = (features * weights[:, None]).sum(dim=0) / w_sum
    centered = features - mean
    cov = (centered * weights[:, None]).T @ centered / w_sum
    return mean, cov


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)

    config = load_config(args.config)
    # 統計計算時は augmentation を無効化
    config["dataset"]["train_aug"] = False

    dataset_name = config["dataset"]["name"]
    backbone = config["regressor"]["config"]["backbone"]
    target_gender = config["dataset"]["config"].get("gender")
    source_gender = "male" if target_gender == "female" else "female"
    target_keys = config["dataset"]["config"].get("target", [])
    if isinstance(target_keys, str):
        target_keys = [target_keys]
    if not target_keys:
        raise RuntimeError("dataset.config.target is required to know label dimension.")
    label_dim = len(target_keys)

    # 事前に保存された特徴量をロード
    feature_path = Path(args.output) / "train_features" / dataset_name
    if source_gender is not None:
        feature_path = feature_path / source_gender
    feature_path = feature_path / f"{backbone}.pt"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    feat_labels = torch.load(feature_path, map_location="cpu")
    if feat_labels.ndim != 2 or feat_labels.shape[1] <= label_dim:
        raise RuntimeError(f"Unexpected feature tensor shape: {feat_labels.shape}")
    src_feat = feat_labels[:, :-label_dim]
    src_label = feat_labels[:, -label_dim:]

    # ターゲットシーケンスのインデックスを決める（val ベース）
    _, val_ds = get_datasets(config)
    base_ds = val_ds
    while hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset  # type: ignore
    if not hasattr(base_ds, "metadata"):
        raise RuntimeError("Base dataset lacks metadata for person filtering.")

    persons = np.array(base_ds.metadata["person"])  # type: ignore
    if not args.target_person:
        raise RuntimeError("Specify --target-person.")
    mask = np.isin(persons, args.target_person)
    target_base_indices = np.nonzero(mask)[0]

    if target_base_indices.size == 0:
        raise RuntimeError("No target samples found for specified persons/indices.")

    # metadata からラベルを直接取得
    target_labels_list = []
    for idx in target_base_indices:
        row = base_ds.metadata.iloc[int(idx)] # type: ignore
        target_labels_list.append([float(row[k]) for k in target_keys])
    tgt_label = torch.tensor(target_labels_list, dtype=torch.float32)

    # densratio は numpy を受け取る
    ratio = densratio(src_label.numpy(), tgt_label.numpy(), alpha=1e-2, kernel_num=200) # type: ignore
    w = torch.from_numpy(ratio.compute_density_ratio(src_label.numpy())).float()
    print(f"Weight stats: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")

    mean, cov = weighted_mean_cov(src_feat, w)
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)

    # 保存先
    stat_dir = Path(args.output) / "stats" / dataset_name
    if source_gender is not None:
        stat_dir = stat_dir / source_gender
    stat_dir.mkdir(parents=True, exist_ok=True)
    out_path = stat_dir / f"{backbone}_{args.target_person}.pt"

    torch.save(
        {
            "mean": mean.cpu(),
            "basis": torch.from_numpy(eigvecs).float(),
            "eigvals": torch.from_numpy(eigvals).float(),
            "weights": w,
        },
        out_path,
    )
    print(f"Saved weighted stats to {out_path}")


if __name__ == "__main__":
    main()
