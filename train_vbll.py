import json
import re
from pathlib import Path
from pprint import pprint
from typing import Any
import copy
import sys
import argparse

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ignite.engine import Engine, Events
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.handlers.param_scheduler import (CosineAnnealingScheduler,
                                             create_lr_scheduler_with_warmup)
import wandb


from models.arch import create_vbll_head
from trainer import VBLLTrainer
from evaluation.evaluator import VBLLEvaluator
from handlers import EvaluationAccumulator, EvaluationRunner
from utils.seed import fix_seed



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VBLL heads directly from cached encoder features.")
    parser.add_argument("-c", required=True, help="config path")
    parser.add_argument("-o", required=True, default="outputs",
                        help="directory to store VBLL checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


def feature_path(config: dict[str, Any], split: str) -> Path:
    custom = config.get("features", {}).get(split)
    if custom is not None:
        return Path(custom)

    dataset = config["dataset"]["name"]
    backbone = config["regressor"]["config"]["backbone"]
    subdir = "train_features" if split == "train" else "valid_features"
    return Path("models", subdir, dataset, f"{backbone}.pt")


def get_feature_dim(config: dict[str, Any]) -> int:
    try:
        feat_dim = int(config["regressor"]["config"]["feature_dim"])
    except KeyError as e:
        raise KeyError("config.regressor.config.feature_dim is required for train_vbll") from e
    
    return feat_dim


def load_feature_bundle(config: dict[str, Any],
                        split: str) -> tuple[Tensor, Tensor]:
    path = feature_path(config, split)

    payload = torch.load(path, map_location="cpu")

    data = payload.float()

    feature_dim = get_feature_dim(config)

    feats = data[:, :feature_dim]
    targets = data[:, feature_dim:]
    return feats, targets


def find_regressor_checkpoint(config: dict[str, Any]) -> Path | None:
    reg_cfg = config.get("regressor", {})
    if (ckpt := reg_cfg.get("source")) is not None:
        return Path(ckpt)

    dataset_name = (config.get("dataset") or {}).get("name")
    backbone = (reg_cfg.get("config") or {}).get("backbone")
    if dataset_name is None or backbone is None:
        return None

    search_dir = Path("models", "weights", dataset_name)
    ds_cfg = (config.get("dataset") or {}).get("config") or {}
    gender = ds_cfg.get("gender")
    if gender:
        search_dir = search_dir / gender

    candidates = [
        p for p in search_dir.glob(f"{backbone}_model_*.pt")
        if "-vbll" not in p.stem
    ]
    if not candidates:
        return None

    def _step_key(path: Path) -> tuple[int, float]:
        match = re.search(r"_model_(\d+)", path.stem)
        step = int(match.group(1)) if match else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (step, mtime)

    return max(candidates, key=_step_key)


def load_linear_head_weight(config: dict[str, Any]) -> tuple[Tensor | None, Path | None]:
    ckpt = find_regressor_checkpoint(config)
    if ckpt is None:
        print("train_vbll: base regressor checkpoint not found; use default VBLL init.")
        return None, None

    state = torch.load(ckpt, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    weight = state_dict.get("regressor.weight")
    if weight is None:
        print(f"train_vbll: 'regressor.weight' not found in {ckpt}; use default VBLL init.")
        return None, ckpt

    return weight, ckpt


def build_loader(dataset: TensorDataset,
                 split: str,
                 config: dict[str, Any]) -> DataLoader:
    if dataset is None:
        raise ValueError("Dataset is required.")

    loader_cfg = dict(config.get(f"{split}_dataloader", {}))
    loader_cfg.setdefault("num_workers", 4)
    loader_cfg.setdefault("pin_memory", torch.cuda.is_available())
    return DataLoader(dataset, **loader_cfg)


def create_optimizer(module: torch.nn.Module,
                     config: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config.get("optimizer")
    if opt_cfg is None:
        raise ValueError("config.optimizer section is required.")
    if opt_cfg.get("param", "all") != "all":
        raise ValueError("Only optimizer.param == 'all' is supported for VBLL head training.")
    opt_cls = getattr(torch.optim, opt_cfg["name"])
    return opt_cls(module.parameters(), **opt_cfg["config"])

def create_scheduler(opt: Optimizer, config: dict[str, Any], iter_per_epoch: int):
    scheduler_config = copy.deepcopy(config["optimizer"]["scheduler"])

    match scheduler_config["type"]:
        case "cos":
            scheduler_config["config"]["cos"]["cycle_size"] = config["epoch"] * iter_per_epoch
            scheduler = CosineAnnealingScheduler(
                opt, **scheduler_config["config"]["cos"])
        case "warmup_cos":
            wd = scheduler_config["config"]["warmup"]["warmup_duration"]
            scheduler_config["config"]["cos"]["cycle_size"] = (
                config["epoch"] - wd + 1) * iter_per_epoch
            cos_sc = CosineAnnealingScheduler(
                opt, **scheduler_config["config"]["cos"])

            scheduler_config["config"]["warmup"]["warmup_duration"] *= iter_per_epoch
            scheduler = create_lr_scheduler_with_warmup(
                cos_sc, **scheduler_config["config"]["warmup"])

        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config!r}")

    return scheduler


def main(args) -> None:
    config = load_config(args.c)
    pprint(config)
    sys.stdout.flush()

    out_dir = Path(args.o, config["dataset"]["name"], config["regressor"]["config"]["backbone"]+"-vbll")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    wandb.init(
        project="tta-reg-train-vbll",
        name=out_dir.name,
        config=config,
        dir=out_dir,
    )

    fix_seed(args.seed)

    train_feats, train_targets = load_feature_bundle(config, "train")
    # print("train_feats (first 5 rows):", train_feats[:5])
    # print("train_targets (first 5 rows):", train_targets[:5])

    train_dataset = TensorDataset(train_feats, train_targets)

    val_feats, val_targets = load_feature_bundle(config, "val")
    val_dataset = TensorDataset(val_feats, val_targets)

    init_weight, init_ckpt = load_linear_head_weight(config)

    head = create_vbll_head(
        in_features=train_feats.shape[1],
        out_features=train_targets.shape[1],
        trainset_size=len(train_dataset),
        init_weight=init_weight,
    ).cuda()
    if init_weight is not None and init_ckpt is not None:
        print(f"VBLL head: initialized W_mean from linear regressor at {init_ckpt}")

    optimizer = create_optimizer(head, config)

    train_loader = build_loader(train_dataset, "train", config)
    val_loader = build_loader(val_dataset, "val", config)

    trainer = VBLLTrainer(head, optimizer)
    evaluator = VBLLEvaluator(head)

    train_ev_logger = EvaluationAccumulator()
    train_ev_runner = EvaluationRunner(
        trainer, train_loader, "train_" + config["dataset"]["name"], train_ev_logger, run_evaluator=False)
    val_ev_logger = EvaluationAccumulator()
    val_ev_runner = EvaluationRunner(
        evaluator, val_loader, "val_" + config["dataset"]["name"], val_ev_logger)

    if "scheduler" in config["optimizer"]:
        scheduler = create_scheduler(optimizer, config, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_ev_runner)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, val_ev_runner)

    model_dir = Path("models/weights", config["dataset"]["name"])
    trainer.add_event_handler(Events.COMPLETED,
                              ModelCheckpoint(dirname=model_dir, require_empty=False,
                                              filename_prefix=config["regressor"]["config"]["backbone"]+"-vbll"),
                              {"model": head})

    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: val_ev_logger.get_dataframe().to_csv(str(out_dir / "val_metrics.csv"), index=False))
    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: train_ev_logger.get_dataframe().to_csv(str(out_dir / "train_metrics.csv"), index=False))
    trainer.run(train_loader, max_epochs=config["epoch"])


    wandb.finish()
    print("done")


if __name__ == "__main__":
    parse_args()
