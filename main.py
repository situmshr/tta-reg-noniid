from typing import Any
from pprint import pprint
import json
from pathlib import Path
import itertools

import yaml

import torch
import vbll
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

from utils.seed import fix_seed
from models.arch import create_regressor, Regressor, extract_bn_layers, extract_gn_layers
from data import get_datasets
from evaluation.evaluator import RegressionEvaluator
from methods import VarianceMinimizationEM


def load_vbll_head(config: dict[str, Any], regressor: Regressor) -> torch.nn.Module:
    vbll_cfg = config.get("tta", {}).get("vbll_head", {}) or {}
    ckpt_path = vbll_cfg.get("path")
    if ckpt_path is None:
        raise ValueError("vbll_head path is not specified in config")
    ckpt = Path(ckpt_path)

    state_dict = torch.load(ckpt, map_location="cpu")

    reg_layer = regressor.get_regressor()
    in_features = vbll_cfg.get("in_features")
    out_features = vbll_cfg.get("out_features")


    reg_weight = vbll_cfg.get("regularization_weight")
    if reg_weight is None:
        trainset_size = vbll_cfg.get("trainset_size")
        reg_weight = 1.0 / trainset_size if trainset_size else 1.0

    prior_scale = vbll_cfg.get("prior_scale", 1.0)
    wishart_scale = vbll_cfg.get("wishart_scale", 0.1)

    vbll_head = vbll.Regression(
        in_features=in_features,
        out_features=out_features,
        regularization_weight=reg_weight,
        prior_scale=prior_scale,
        wishart_scale=wishart_scale,
    )
    vbll_head.load_state_dict(state_dict)
    vbll_head.eval()
    for param in vbll_head.parameters():
        param.requires_grad_(False)
    return vbll_head




def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    fix_seed(args.seed)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)

    tta_cfg = config.get("tta") or {}
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    val_corruption = dataset_cfg.get("val_corruption") or {}
    severity = val_corruption.get("severity")
    dataset_dir = f"{dataset_name}-{severity}" if severity is not None else dataset_name
    backbone = config["regressor"]["config"]["backbone"]
    method = tta_cfg.get("method")
    base_mode = method is None or method == "base"
    method_name = method if method is not None else "base"

    output_dir = Path(args.o, dataset_dir, f"{backbone}-{method_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(output_dir, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    engine = None
    if not base_mode:
        if method != "vm":
            raise ValueError(f"Unsupported TTA method: {method!r}. Only 'vm' or 'base' are supported.")
        opt = create_optimizer(regressor, config)
        vbll_head = load_vbll_head(config, regressor)
        variance_weight = tta_cfg.get("variance_weight", 1.0)

        engine = VarianceMinimizationEM(
            regressor,
            opt,
            vbll_head=vbll_head,
            variance_weight=variance_weight,
            **config["trainer"],
        )

        if args.save:
            engine.add_event_handler(
                Events.COMPLETED,
                ModelCheckpoint(output_dir, "adapted", require_empty=False),
                {"regressor": regressor},
            )
    elif args.save:
        torch.save(regressor.state_dict(), Path(output_dir, "regressor.pt"))

    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])

    if engine is not None:
        engine.run(val_dl)
    reg_evaluator.run(val_dl)

    metrics = {
        "iteration": engine.state.iteration if engine is not None else 0,
        "online": engine.state.metrics if engine is not None else {},
        "offline": reg_evaluator.state.metrics
    }
    pprint(metrics)
    with Path(output_dir, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def create_optimizer(net: Regressor, config: dict[str, Any]) -> torch.optim.Optimizer:
    match config["optimizer"]["param"]:
        case "all":
            params = net.parameters()
        case "fe":
            params = net.get_feature_extractor().parameters()
        case "fe_bn":
            bn_layers = extract_bn_layers(net.get_feature_extractor())
            params = itertools.chain.from_iterable(
                l.parameters() for l in bn_layers
            )
        case "fe_gn":
            print("Using GN layers for optimization")
            gn_layers = extract_gn_layers(net.get_feature_extractor())
            params = itertools.chain.from_iterable(
                l.parameters() for l in gn_layers
            )
        case _ as p:
            raise ValueError(f"Invalid param: {p!r}")

    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        params, **config["optimizer"]["config"])
    return opt


if __name__ == "__main__":
    parse_args()
