from typing import Any
import json
from pathlib import Path
from pprint import pprint
import copy
import sys
import argparse
import os

import yaml

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from ignite.engine.events import Events
from ignite.handlers.param_scheduler import (CosineAnnealingScheduler,
                                             create_lr_scheduler_with_warmup)
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.metrics import Average
import wandb

from data import get_datasets
from models.arch import create_regressor, Regressor
from trainer import RegressionTrainer
from evaluation.evaluator import RegressionEvaluator

from handlers import EvaluationAccumulator, EvaluationRunner, make_run_val_epoch
from utils.atloc_loss import AtLocCriterion, AtLocPlusCriterion
from utils.seed import fix_seed

DISPLAY = "DISPLAY" in os.environ
if not DISPLAY:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_loss(config: dict[str, Any]) -> nn.Module | None:
    loss_cfg = config.get("loss") or {}
    name = loss_cfg.get("name")
    if not name:
        return None

    if name == "AtLocPlusCriterion":
        return AtLocPlusCriterion(
            sax=loss_cfg.get("sax", 0.0),
            saq=loss_cfg.get("beta", loss_cfg.get("saq", -3.0)),
            srx=loss_cfg.get("srx", 0.0),
            srq=loss_cfg.get("gamma", loss_cfg.get("srq", -3.0)),
            learn_beta=loss_cfg.get("learn_beta", True),
            learn_gamma=loss_cfg.get("learn_gamma", True),
        )
    if name == "AtLocCriterion":
        return AtLocCriterion(
            sax=loss_cfg.get("sax", 0.0),
            saq=loss_cfg.get("beta", loss_cfg.get("saq", -3.0)),
            learn_beta=loss_cfg.get("learn_beta", True),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, default="outputs", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)
    sys.stdout.flush()

    # save config in structured (dataset/scene) directory
    dataset_name = config["dataset"]["name"]
    scene = config["dataset"]["config"].get("scene")
    run_dir = Path(args.o) / dataset_name
    if scene is not None:
        run_dir = run_dir / scene
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    wandb.init(
        project="tta-reg-train-source",
        name=run_dir.name,
        config=config,
        dir=str(run_dir),
    )

    fix_seed(args.seed)

    net = create_regressor(config).cuda()

    loss_fn = create_loss(config)
    if loss_fn is not None:
        loss_fn = loss_fn.cuda()

    opt = create_optimizer(net, config, loss_fn)

    train_ds, val_ds = get_datasets(config)
    train_dl = DataLoader(train_ds, **config["train_dataloader"], num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, **config["val_dataloader"], num_workers=4, pin_memory=True)

    trainer = RegressionTrainer(net, opt, loss_fn=loss_fn, **config["trainer"])

    if "scheduler" in config["optimizer"]:
        scheduler = create_scheduler(opt, config, len(train_dl))
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    train_ev_logger = EvaluationAccumulator()
    train_ev_runner = EvaluationRunner(
        trainer, train_dl, "train_" + dataset_name, train_ev_logger, run_evaluator=False
    )

    ds_targets = (config.get("dataset", {}).get("config", {}) or {}).get("target")
    eval_cfg = dict(config["evaluator"])
    eval_cfg.setdefault("val_dataset", val_ds)
    if ds_targets is not None:
        eval_cfg.setdefault("target_names", ds_targets)
    evaluator = RegressionEvaluator(net, **eval_cfg)

    val_ev_logger = EvaluationAccumulator()
    val_ev_runner = EvaluationRunner(
        evaluator, val_dl, "val_" + dataset_name, val_ev_logger)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_ev_runner)

    is_four_seasons = dataset_name == "4seasons"

    run_val_epoch = make_run_val_epoch(
        val_ev_runner,
        is_four_seasons,
        net,
        val_dl,
        config["dataset"]["config"],
        run_dir,
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_val_epoch)

    model_dir = Path("models/weights", dataset_name)
    gender = config["dataset"]["config"].get("gender")
    if gender is not None:
        model_dir = model_dir / gender
    if scene is not None:
        model_dir = model_dir / scene

    trainer.add_event_handler(Events.COMPLETED,
                              ModelCheckpoint(dirname=model_dir, require_empty=False,
                                              filename_prefix=config["regressor"]["config"]["backbone"]),
                              {"model": net})

    p = run_dir
    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: val_ev_logger.get_dataframe().to_csv(str(p / "val_metrics.csv"), index=False))
    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: train_ev_logger.get_dataframe().to_csv(str(p / "train_metrics.csv"), index=False))
    trainer.run(train_dl, max_epochs=config["epoch"])

    wandb.finish()
    print("done")


def create_optimizer(
    net: Regressor,
    config: dict[str, Any],
    loss_fn: nn.Module | None = None,
) -> Optimizer:
    match p := config["optimizer"]["param"]:
        case "all":
            param = net.parameters()
        case _:
            raise ValueError(f"Invalid param: {p!r}")

    print("param:", p)

    param_groups = [{"params": param}]
    if loss_fn is not None:
        loss_params = []
        for attr in ("sax", "saq", "srx", "srq"):
            p_attr = getattr(loss_fn, attr, None)
            if isinstance(p_attr, nn.Parameter) and p_attr.requires_grad:
                loss_params.append(p_attr)
        if loss_params:
            param_groups.append({"params": loss_params})

    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        param_groups, **config["optimizer"]["config"])
    return opt


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


if __name__ == "__main__":
    parse_args()
