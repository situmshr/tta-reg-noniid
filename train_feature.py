from dataclasses import dataclass, InitVar

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine

import yaml

from data import get_datasets
from models.arch import create_regressor, Regressor
from utils.config_process import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract encoder features following a config file.")
    parser.add_argument("-c", required=True, help="config path")
    parser.add_argument("-o", default="models/train_features",
                        help="directory to store features")
    return parser.parse_args()


def build_loader(config: dict) -> DataLoader:
    train_ds, _ = get_datasets(config)
    dataset = train_ds
    return DataLoader(dataset, **config["dataloader"], num_workers=4, pin_memory=True)


@dataclass
class FeatureCalculator(Engine):
    regressor: Regressor
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        # Register inference to the loop
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
    def inference(self, engine: Engine, batch: tuple[torch.Tensor, torch.Tensor]):
        self.regressor.eval()

        x, y = batch
        x = x.cuda()

        feat = self.regressor_feature(x)    # (B,D)
        self._features.append(feat.cpu())
        self._labels.append(y.cpu())

    def get_feats(self, ddof: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(self._features)
        labels = torch.cat(self._labels)

        return features, labels


def main() -> None:
    args = parse_args()
    config = load_config(args.c)

    loader = build_loader(config)
    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    calculator = FeatureCalculator(regressor, **config["calculator"])
    calculator.run(loader)
    features, labels = calculator.get_feats()
    labels = labels.view(labels.size(0), -1)
    feat_labels = torch.cat([features, labels], dim=1)

    out_dir = Path(args.o, config["dataset"]["name"])
    if config["dataset"]["config"].get("gender") is not None:
        out_dir = out_dir / config["dataset"]["config"]["gender"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config['regressor']['config']['backbone']}.pt"
    torch.save(feat_labels, out_path)
    print(f"saved {feat_labels.shape} -> {out_path}")


if __name__ == "__main__":
    main()
