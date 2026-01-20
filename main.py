from typing import Any
from pprint import pprint
import json
from pathlib import Path
import itertools

import yaml

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events

from utils.seed import fix_seed
from utils.viz_label_stream import visualize_label_stream
from models.arch import create_regressor, Regressor, extract_bn_layers, extract_gn_layers
from data import get_datasets, get_non_iid_dataset
from data.image_utils import ImageSubset
from evaluation.evaluator import RegressionEvaluator
from evaluation.four_seasons_eval import (
    collate_first_two,
    evaluate_four_seasons,
    FourSeasonsOnlineTracker,
)
from methods import (
    SSA,
    RS_SSA,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")
    parser.add_argument(
        "--viz-label-stream",
        action="store_true",
        help="visualize the label order emitted by val_dl",
    )

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
    config["seed"] = args.seed
    pprint(config)

    tta_cfg = config.get("tta") or {}
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    data_stream_cfg = dataset_cfg.get("data_stream") or {}
    stream_type = (data_stream_cfg.get("type") or "iid").lower()
    stream_label = stream_type
    non_iid_kwargs: dict[str, Any] | None = None
    if stream_type == "non_iid":
        cycles = data_stream_cfg.get("cycles")
        if cycles is None:
            cycles = data_stream_cfg.get("period", 1)

        sigma = data_stream_cfg.get("sigma")
        if sigma is None:
            sigma = data_stream_cfg.get("sigma_label")
        if sigma is None:
            sigma = data_stream_cfg.get("beta", 1.0)

        non_iid_kwargs = {
            "mode": data_stream_cfg.get("mode", "linear"),
            "sigma": sigma,
            "cycles": cycles,
            "length": data_stream_cfg.get("length"),
            "seed": data_stream_cfg.get("seed", args.seed),
        }
        stream_parts = ["non_iid", non_iid_kwargs["mode"]]
        if sigma is not None:
            stream_parts.append(f"sigma{sigma}")
        if cycles not in (None, 1):
            stream_parts.append(f"cycles{cycles}")
        stream_label = "-".join(map(str, stream_parts))
    elif stream_type != "iid":
        raise ValueError(
            f"Unsupported data stream type: {stream_type!r}. Use 'iid' or 'non_iid'."
        )
    val_corruption = dataset_cfg.get("val_corruption") or {}
    severity = val_corruption.get("severity")
    dataset_dir = f"{dataset_name}-{severity}" if severity is not None else dataset_name
    corruption_name = val_corruption.get("corruption_name") or val_corruption.get("name")
    backbone = config["regressor"]["config"]["backbone"]
    raw_method = tta_cfg.get("method")
    method_key = "base" if raw_method is None else str(raw_method).lower()
    method_name = raw_method if raw_method is not None else "base"

    if corruption_name is not None and severity is not None:
        dataset_dir = f"{dataset_name}-{corruption_name}-{severity}"
    seed_label = f"seed{args.seed}"
    if dataset_name == "utkface":
        output_dir = Path(args.o, stream_label, dataset_dir, f"{backbone}-{method_name}", seed_label)
    elif dataset_name == "4seasons":
        scene = config["dataset"]["config"].get("scene", "unknown")
        seq_name = config["dataset"]["config"].get("seqs", "unknown")[0]
        output_dir = Path(args.o, dataset_dir, scene, seq_name, f"{backbone}-{method_name}", seed_label)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(output_dir, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)

    if stream_type == "non_iid":
        print(f"Applying non-iid data stream: {non_iid_kwargs}")
        val_ds = get_non_iid_dataset(val_ds, **(non_iid_kwargs or {}))

    if dataset_name == "4seasons":
        val_dl = DataLoader(val_ds, **config["adapt_dataloader"], collate_fn=collate_first_two)
    else:
        val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    if args.viz_label_stream:
        visualize_label_stream(val_dl, Path(output_dir, "val_label_stream.png"))

    trainer_cfg = dict(config.get("trainer", {}))
    trainer_cfg.setdefault("val_dataset", val_ds)
    eval_cfg = dict(config.get("evaluator", {}))
    eval_cfg.setdefault("val_dataset", val_ds)
    # Optional: also pass explicit target names from dataset config for clarity
    ds_targets = (dataset_cfg.get("config", {}) or {}).get("target")
    if ds_targets is not None:
        eval_cfg.setdefault("target_names", ds_targets)
        trainer_cfg.setdefault("target_names", ds_targets)

    engine = None

    if method_key == "base":
        pass
    elif method_key == "ssa":
        opt = create_optimizer(regressor, config)
        ssa_kwargs = {
            "pc_config": tta_cfg.get("pc_config"),
            "loss_config": tta_cfg.get("loss_config"),
            "weight_bias": tta_cfg.get("weight_bias", 1e-6),
            "weight_exp": tta_cfg.get("weight_exp", 1.0),
        }
        engine = SSA(
            regressor,
            opt,
            **trainer_cfg,
            **ssa_kwargs,
        )
    elif method_key == "rs_ssa":
        opt = create_optimizer(regressor, config)
        rs_kwargs = {
            "pc_config": tta_cfg.get("pc_config"),
            "buffer_size": tta_cfg.get("buffer_size", 32),
            "min_buffer_size": tta_cfg.get("min_buffer_size", 1),
            "ssa_weight": tta_cfg.get("ssa_weight", 1.0),
            "ema_alpha": tta_cfg.get("ema_alpha", 0.1),
            "ema_momentum": tta_cfg.get("ema_momentum", 0.99),
        }
        engine = RS_SSA(
            regressor,
            opt,
            **trainer_cfg,
            **rs_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported TTA method: {raw_method!r}. Supported: 'base', 'vm', 'ssa', 'wssa', 'adabn', 'ada_ssa', 'rs_ssa'."
        )

    if args.save:
        output_root = Path(args.o)
        rel_path = output_dir.relative_to(output_root)
        rel_parts = rel_path.parts
        if len(rel_parts) >= 2:
            save_subdir = Path(*rel_parts[:-2])
        else:
            save_subdir = Path()
        save_dir = Path("models", "tta_weights", save_subdir)
        save_path = save_dir / f"{backbone}_{method_key}.pt"
        save_dir.mkdir(parents=True, exist_ok=True)
        if engine is None:
            torch.save(regressor.state_dict(), save_path)
        else:
            def save_regressor(_) -> None:
                torch.save(regressor.state_dict(), save_path)

            engine.add_event_handler(Events.COMPLETED, save_regressor)

    is_four_seasons = dataset_name == "4seasons"
    reg_evaluator = None if is_four_seasons else RegressionEvaluator(regressor, **eval_cfg)

    tracker = None
    if is_four_seasons and engine is not None:
        tracker = FourSeasonsOnlineTracker(config["dataset"]["config"])
        engine.add_event_handler(Events.ITERATION_COMPLETED, tracker.update)

    if engine is not None:
        engine.run(val_dl)

    online_eval_metrics = None
    if is_four_seasons:
        fig_dir = output_dir / "figures"
        offline_fig = fig_dir / "trajectory.png"
        online_fig = fig_dir / "trajectory_online.png"
        # online-style eval during adaptation (uses tracker if available)
        if tracker is not None:
            online_eval_metrics = tracker.compute(fig_path=online_fig)
        # else:
        #     online_eval_metrics = evaluate_four_seasons_online(
        #         regressor, val_dl, config["dataset"]["config"], fig_path=online_fig
        #     )
        # offline-style eval (full pass, aggregate)
        offline_metrics = evaluate_four_seasons(
            regressor, val_dl, config["dataset"]["config"], fig_path=offline_fig
        )
    else:
        reg_evaluator.run(val_dl) # type: ignore
        offline_metrics = reg_evaluator.state.metrics # type: ignore

    metrics = {
        "iteration": engine.state.iteration if engine is not None else 0,
        "online": engine.state.metrics if engine is not None else {},
        "offline": offline_metrics
    }
    if online_eval_metrics is not None:
        metrics["online_eval"] = online_eval_metrics
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
