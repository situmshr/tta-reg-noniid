import argparse
from typing import Any
from pprint import pprint
import json
from pathlib import Path
import itertools

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events

from utils.seed import fix_seed, preserve_rng_state
from utils.viz_label_stream import visualize_label_stream
from utils.config_process import (
    load_config,
    build_stream_label,
    resolve_tta_dir,
    resolve_save_model_path,
    save_config,
    save_metrics,
)
from models.arch import create_regressor, Regressor, extract_bn_layers, extract_gn_layers
from data import get_datasets, get_non_iid_dataset
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


def _setup_model_saving(engine, regressor, save_path: Path) -> None:
    """Save model immediately (no engine) or register on COMPLETED."""
    if engine is None:
        torch.save(regressor.state_dict(), save_path)
    else:
        def _save(_) -> None:
            torch.save(regressor.state_dict(), save_path)
        engine.add_event_handler(Events.COMPLETED, _save)


def _run_evaluation(
    regressor,
    engine,
    val_dl: DataLoader,
    config: dict,
    eval_cfg: dict,
    output_dir: Path,
) -> dict:
    """Run offline/online evaluation and return metrics dict."""
    dataset_name = config["dataset"]["name"]
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
        if tracker is not None:
            online_eval_metrics = tracker.compute(fig_path=fig_dir / "trajectory_online.png")
        offline_metrics = evaluate_four_seasons(
            regressor, val_dl, config["dataset"]["config"],
            fig_path=fig_dir / "trajectory.png",
        )
    else:
        reg_evaluator.run(val_dl)  # type: ignore
        offline_metrics = reg_evaluator.state.metrics  # type: ignore

    metrics: dict[str, Any] = {
        "iteration": engine.state.iteration if engine is not None else 0,
        "online": engine.state.metrics if engine is not None else {},
        "offline": offline_metrics,
    }
    if online_eval_metrics is not None:
        metrics["online_eval"] = online_eval_metrics
    return metrics


def main(args):
    config = load_config(args.c)
    config["seed"] = args.seed
    pprint(config)

    fix_seed(args.seed)

    tta_cfg = config.get("tta") or {}
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    data_stream_cfg = dataset_cfg.get("data_stream") or {}

    # --- Path resolution ---
    stream_type = (data_stream_cfg.get("type") or "iid").lower()
    stream_label, non_iid_kwargs = build_stream_label(data_stream_cfg, args.seed)
    output_dir = resolve_tta_dir(config, args.o, stream_label, args.seed)
    save_config(config, output_dir)

    raw_method = tta_cfg.get("method")
    method_key = "base" if raw_method is None else str(raw_method).lower()
    backbone = config["regressor"]["config"]["backbone"]

    # --- Model & data ---
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
        with preserve_rng_state():
            visualize_label_stream(val_dl, output_dir / "val_label_stream.png")

    trainer_cfg = dict(config.get("trainer", {}))
    trainer_cfg.setdefault("val_dataset", val_ds)
    eval_cfg = dict(config.get("evaluator", {}))
    eval_cfg.setdefault("val_dataset", val_ds)

    # --- Engine construction ---
    engine = None

    if method_key == "base":
        pass
    elif method_key == "ssa":
        opt = create_optimizer(regressor, config)
        engine = SSA(
            regressor, opt, **trainer_cfg,
            pc_config=tta_cfg.get("pc_config"),
            loss_config=tta_cfg.get("loss_config"),
            weight_bias=tta_cfg.get("weight_bias", 1.0),
            weight_exp=tta_cfg.get("weight_exp", 1.0),
        )
    elif method_key == "rs_ssa":
        opt = create_optimizer(regressor, config)
        engine = RS_SSA(
            regressor, opt, **trainer_cfg,
            pc_config=tta_cfg.get("pc_config"),
            buffer_size=tta_cfg.get("buffer_size", 64),
            min_buffer_size=tta_cfg.get("min_buffer_size", 64),
            ssa_weight=tta_cfg.get("ssa_weight", 1.0),
            ema_alpha=tta_cfg.get("ema_alpha", 0.1),
            ema_momentum=tta_cfg.get("ema_momentum", 0.99),
        )
    else:
        raise ValueError(
            f"Unsupported TTA method: {raw_method!r}. "
            "Supported: 'base', 'ssa', 'rs_ssa'."
        )

    # --- Model saving ---
    if args.save:
        save_path = resolve_save_model_path(output_dir, args.o, backbone, method_key)
        _setup_model_saving(engine, regressor, save_path)

    # --- Evaluation & metrics ---
    metrics = _run_evaluation(regressor, engine, val_dl, config, eval_cfg, output_dir)
    pprint(metrics)
    save_metrics(metrics, output_dir)


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
