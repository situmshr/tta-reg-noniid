#!/usr/bin/env python3
"""Save encoder features for a specified model and dataset config."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from pathlib import Path
from pprint import pprint
import argparse
import json

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ignite.engine import Engine

import yaml

from data import get_datasets
from models.arch import create_regressor, Regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save encoder features from a model on a specified dataset split."
    )
    parser.add_argument("-c", "--config", required=True, help="Config file (dataset + model).")
    parser.add_argument(
        "-o",
        "--output",
        default="models",
        help="Output root directory (features saved under <split>_features/<dataset>/...).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Override model checkpoint path (default: config regressor.source).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to extract features from.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers.")
    args = parser.parse_args()
    pprint(vars(args))
    return args


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


@dataclass
class FeatureExtractor(Engine):
    regressor: Regressor
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.inference)
        if compile_model is None:
            self.regressor_feature = self.regressor.feature
        else:
            try:
                self.regressor_feature = torch.compile(
                    self.regressor.feature, **compile_model
                )
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.regressor_feature = self.regressor.feature
        self.reset()

    def reset(self) -> None:
        self._features = []
        self._labels = []

    @torch.no_grad()
    def inference(self, engine: Engine, batch: tuple[Tensor, Tensor]):
        self.regressor.eval()
        x, y = batch[0], batch[1]
        x = x.cuda()
        feat = self.regressor_feature(x)
        self._features.append(feat.cpu())
        self._labels.append(y.cpu())

    def get_features(self) -> tuple[Tensor, Tensor]:
        if not self._features:
            raise RuntimeError("No features collected.")
        features = torch.cat(self._features)
        labels = torch.cat(self._labels)
        return features, labels


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["dataset"]["train_aug"] = False

    train_ds, val_ds = get_datasets(config)
    dataset = train_ds if args.split == "train" else val_ds
    loader = DataLoader(
        dataset,
        **config["dataloader"],
        num_workers=args.num_workers,
        pin_memory=True,
    )

    regressor = create_regressor(config).cuda()
    weights_path = args.weights or config["regressor"]["source"]
    regressor.load_state_dict(torch.load(weights_path))
    print(f"load regressor: {weights_path}")

    extractor = FeatureExtractor(regressor, **config["calculator"])
    extractor.run(loader)
    features, labels = extractor.get_features()
    labels = labels.view(labels.size(0), -1)
    feat_labels = torch.cat([features, labels], dim=1)

    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "dataset")
    feature_dir = Path(args.output, f"{args.split}_features", dataset_name)
    extra_cfg = dataset_cfg.get("config", {}) or {}
    if extra_cfg.get("gender") is not None:
        feature_dir = feature_dir / extra_cfg["gender"]
    if extra_cfg.get("scene") is not None:
        feature_dir = feature_dir / extra_cfg["scene"]
    feature_dir.mkdir(parents=True, exist_ok=True)

    backbone = config["regressor"]["config"]["backbone"]
    out_path = feature_dir / f"{backbone}.pt"
    torch.save(feat_labels, out_path)
    print(f"saved {feat_labels.shape} -> {out_path}")


if __name__ == "__main__":
    main()
