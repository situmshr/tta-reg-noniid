from typing import Any
from pathlib import Path
from pprint import pprint
import copy
import argparse
import os

import torch
import matplotlib
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from ignite.engine.events import Events
from ignite.handlers.param_scheduler import (CosineAnnealingScheduler,
                                             create_lr_scheduler_with_warmup)
from ignite.handlers.checkpoint import ModelCheckpoint
import wandb

from data import get_datasets
from models.arch import create_regressor, Regressor
from trainer import RegressionTrainer
from evaluation.evaluator import RegressionEvaluator

from handlers import EvaluationAccumulator, EvaluationRunner, make_run_val_epoch
from utils.atloc_loss import AtLocCriterion, AtLocPlusCriterion
from utils.seed import fix_seed
from utils.config_process import load_config, save_config, resolve_path_from_config

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
    config = load_config(args.c)
    run_dir, train_dir, val_dir, model_dir = resolve_path_from_config(config, args.o)
    save_config(config, args.o)

    fix_seed(args.seed)

    wandb.init(
        project="tta-reg-train-source",
        name=run_dir.name,
        config=config,
        dir=str(run_dir),
    )

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
        trainer, train_dl, train_dir, train_ev_logger, run_evaluator=False
    )

    eval_cfg = dict(config["evaluator"])
    eval_cfg.setdefault("val_dataset", val_ds)
    evaluator = RegressionEvaluator(net, **eval_cfg)

    val_ev_logger = EvaluationAccumulator()
    val_ev_runner = EvaluationRunner(
        evaluator, val_dl, val_dir, val_ev_logger
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_ev_runner)

    is_four_seasons = config["dataset"]["name"] == "4seasons"

    run_val_epoch = make_run_val_epoch(
        val_ev_runner,
        is_four_seasons,
        net,
        val_dl,
        config["dataset"]["config"],
        run_dir,
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_val_epoch)

    trainer.add_event_handler(Events.COMPLETED,
                              ModelCheckpoint(dirname=model_dir, require_empty=False,
                                              filename_prefix=config["regressor"]["config"]["backbone"]),
                              {"model": net})

    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: val_ev_logger.get_dataframe().to_csv(str(run_dir / "val_metrics.csv"), index=False))
    trainer.add_event_handler(Events.COMPLETED,
                              lambda _: train_ev_logger.get_dataframe().to_csv(str(run_dir / "train_metrics.csv"), index=False))
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
            param_groups.append({"params": loss_params}) # type: ignore

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
