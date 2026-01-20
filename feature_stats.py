from dataclasses import dataclass, InitVar
from pprint import pprint
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ignite.engine import Engine

import numpy as np
import yaml

from data import get_datasets
from models.arch import create_regressor, Regressor


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True,
                        help="config file")
    parser.add_argument("-o", required=True, default="models", help="output directory")
    parser.add_argument("--save_feature", action="store_true")
    parser.add_argument("--validation", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    Path(args.o).mkdir(parents=True, exist_ok=True)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    config["dataset"]["train_aug"] = False
    pprint(config)

    ds = get_datasets(config)[1 if args.validation else 0]
    dl = DataLoader(ds, **config["dataloader"], num_workers=4, pin_memory=True)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load regressor: {p}")

    calc = FeatureStatCalculator(
        regressor, **config["calculator"])
    print("run calculator", flush=True)
    calc.run(dl)

    mean, cov, features, labels = calc.compute_stats()
    labels = labels.view(labels.size(0), -1)
    feat_labels = torch.cat([features, labels], dim=1)
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)

    print(f"rank of cov: {np.linalg.matrix_rank(cov_np)}")

    stat_dir = Path(args.o, "stats", config["dataset"]["name"])
    if config["dataset"]["config"].get("gender") is not None:
        stat_dir = stat_dir / config["dataset"]["config"]["gender"]
    if config["dataset"]["config"].get("scene") is not None:
        stat_dir = stat_dir / config["dataset"]["config"]["scene"]
    stat_dir.mkdir(parents=True, exist_ok=True)
    p = Path(stat_dir, config["regressor"]["config"]["backbone"]+".pt")
    torch.save({
        "mean": mean,
        "basis": torch.from_numpy(eigvecs),
        "eigvals": torch.from_numpy(eigvals)
    }, str(p))

    if args.save_feature:
        feature_dir = Path(args.o, "train_features", config["dataset"]["name"])
        if config["dataset"]["config"].get("gender") is not None:
            feature_dir = feature_dir / config["dataset"]["config"]["gender"]
        if config["dataset"]["config"].get("scene") is not None:
            feature_dir = feature_dir / config["dataset"]["config"]["scene"]
        feature_dir.mkdir(parents=True, exist_ok=True)
        p = Path(feature_dir, config["regressor"]["config"]["backbone"]+".pt")
        torch.save(feat_labels, str(p))
        print("feature saved")


@dataclass
class FeatureStatCalculator(Engine):
    regressor: Regressor
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.inference)

        if compile_model is None:
            self.regressor_feature = self.regressor.feature
        else:
            try:
                self.regressor_feature = torch.compile(
                    self.regressor.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.regressor_feature = self.regressor.feature

        self.reset()

    def reset(self):
        self._features = []
        self._labels = []

    @torch.no_grad()
    def inference(self, engine: Engine, batch: tuple[Tensor, Tensor]):
        self.regressor.eval()

        # Some datasets may return (x, y, *extras); we only need the first two.
        x, y = batch[0], batch[1]
        if x.dim() == 5:
            # Use the center frame to reduce [B,L,C,H,W] -> [B,C,H,W].
            # Exclude non-center frames since they overlap; all samples are still evaluated.
            seq_len = x.size(1)
            center = seq_len // 2
            x = x[:, center]
            if y.dim() == 3 and y.size(1) == seq_len:
                y = y[:, center]
        x = x.cuda()

        feat = self.regressor_feature(x)    # (B,D)
        self._features.append(feat.cpu())
        self._labels.append(y.cpu())

    def compute_stats(self, ddof: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert len(self._features) >= 1, "no data accumulated"

        features = torch.cat(self._features)
        labels = torch.cat(self._labels)
        mean = features.mean(dim=0)
        features_c = features - mean    # (B,D)
        cov = features_c.T @ features_c / (features.shape[0] - ddof)

        return mean, cov, features, labels


if __name__ == "__main__":
    parse_args()
