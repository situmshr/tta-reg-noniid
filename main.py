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
from data import get_datasets, get_note_non_iid_dataset
from evaluation.evaluator import RegressionEvaluator
from methods import (
    VarianceMinimizationEM,
    SignificantSubspaceAlignment,
    WeightedSignificantSubspaceAlignment,
    AdaptiveBatchNorm,
    AdaptiveSSA,
    ER_SSA,
)


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
    pprint(config)

    tta_cfg = config.get("tta") or {}
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    data_stream_cfg = dataset_cfg.get("data_stream") or {}
    base_stream_type = (data_stream_cfg.get("type") or "iid").lower()
    stream_type = base_stream_type
    beta = data_stream_cfg.get("beta")
    if base_stream_type == "non_iid" and beta is not None:
        stream_type = f"non_iid-{beta}"
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
    output_dir = Path(args.o, stream_type, dataset_dir, f"{backbone}-{method_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(output_dir, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)

    if base_stream_type == "non_iid":
        required_keys = ("num_chunks", "beta", "min_chunk_size", "num_bins")
        missing = [k for k in required_keys if k not in data_stream_cfg]
        if missing:
            raise ValueError(
                "Missing non_iid data stream settings: " + ", ".join(missing)
            )

        stream_kwargs = {k: data_stream_cfg[k] for k in required_keys}
        if "seed" in data_stream_cfg:
            stream_kwargs["seed"] = data_stream_cfg["seed"]
        else:
            stream_kwargs["seed"] = args.seed

        print(f"Applying non-iid data stream: {stream_kwargs}")
        val_ds = get_note_non_iid_dataset(val_ds, **stream_kwargs)
    elif stream_type != "iid":
        raise ValueError(
            f"Unsupported data stream type: {stream_type!r}. Use 'iid' or 'non_iid'."
        )
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    if args.viz_label_stream:
        visualize_label_stream(val_dl, Path(output_dir, "val_label_stream.png"))

    

    trainer_cfg = config.get("trainer", {})
    engine = None

    if method_key == "base":
        pass
    elif method_key == "vm":
        opt = create_optimizer(regressor, config)
        vbll_head = load_vbll_head(config, regressor)
        variance_weight = tta_cfg.get("variance_weight", 1.0)

        engine = VarianceMinimizationEM(
            regressor,
            opt,
            vbll_head=vbll_head,
            variance_weight=variance_weight,
            **trainer_cfg,
        )
    elif method_key == "ssa":
        opt = create_optimizer(regressor, config)
        ssa_kwargs = {
            "pc_config": tta_cfg.get("pc_config"),
            "loss_config": tta_cfg.get("loss_config"),
            "weight_bias": tta_cfg.get("weight_bias", 1e-6),
            "weight_exp": tta_cfg.get("weight_exp", 1.0),
        }
        engine = SignificantSubspaceAlignment(
            regressor,
            opt,
            **trainer_cfg,
            **ssa_kwargs,
        )
    elif method_key == "adabn":
        engine = AdaptiveBatchNorm(
            regressor,
            None,
            **trainer_cfg,
        )
    elif method_key == "wssa":
        opt = create_optimizer(regressor, config)
        wssa_kwargs = {
            "pc_config": tta_cfg.get("pc_config"),
            "loss_config": tta_cfg.get("loss_config"),
            "weight_bias": tta_cfg.get("weight_bias", 1e-6),
            "weight_exp": tta_cfg.get("weight_exp", 1.0),
            "temperature": tta_cfg.get("temperature", 1.0),
        }
        engine = WeightedSignificantSubspaceAlignment(
            regressor,
            opt,
            **trainer_cfg,
            **wssa_kwargs,
        )
    elif method_key == "ada_ssa":
        opt = create_optimizer(regressor, config)
        ada_kwargs = {
            "ema_alpha": tta_cfg.get("ema_alpha", 0.1),
            "base_ssa_weight": tta_cfg.get("base_ssa_weight", 0.0),
            "max_ssa_weight": tta_cfg.get("max_ssa_weight", 1.0),
            "ssa_growth_rate": tta_cfg.get("ssa_growth_rate", 0.1),
            "t_threshold": tta_cfg.get("t_threshold", 3.0),
            "alpha_batch": tta_cfg.get("alpha_batch", 0.5),
            "pc_config": tta_cfg.get("pc_config"),
        }
        engine = AdaptiveSSA(
            regressor,
            opt,
            **trainer_cfg,
            **ada_kwargs,
        )
    elif method_key == "er_ssa":
        opt = create_optimizer(regressor, config)
        er_kwargs = {
            "pc_config": tta_cfg.get("pc_config"),
            "buffer_size": tta_cfg.get("buffer_size", 32),
            "min_buffer_size": tta_cfg.get("min_buffer_size", 1),
            "ssa_weight": tta_cfg.get("ssa_weight", 1.0),
            "ema_alpha": tta_cfg.get("ema_alpha", 0.1),
            "main_loss": tta_cfg.get("main_loss", "none"),
        }
        engine = ER_SSA(
            regressor,
            opt,
            **trainer_cfg,
            **er_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported TTA method: {raw_method!r}. Supported: 'base', 'vm', 'ssa', 'wssa', 'adabn', 'ada_ssa', 'er_ssa'."
        )

    if args.save:
        if engine is None:
            torch.save(regressor.state_dict(), Path(output_dir, "regressor.pt"))
        else:
            engine.add_event_handler(
                Events.COMPLETED,
                ModelCheckpoint(output_dir, "adapted", require_empty=False),
                {"regressor": regressor},
            )

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


def visualize_label_stream(dataloader: DataLoader, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - optional dependency
        print(f"matplotlib is required for --viz-label-stream but not found: {err}")
        return

    labels: list[torch.Tensor] = []
    for _, y in dataloader:
        if isinstance(y, torch.Tensor):
            labels.append(y.detach().flatten().cpu())
        else:
            labels.append(torch.as_tensor(y).flatten().cpu())

    if not labels:
        print("No labels available to visualize.")
        return

    label_tensor = torch.cat(labels).float()
    idx = torch.arange(label_tensor.numel())
    label_np = label_tensor.numpy()
    idx_np = idx.numpy()
    bins = min(50, max(10, label_tensor.numel() // 10))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=False,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    axes[0].plot(idx_np, label_np, linewidth=0.8)
    axes[0].set_ylabel("Label")
    axes[0].set_xlabel("Sample index")
    axes[0].set_title("Validation label stream (order of appearance)")
    axes[0].grid(alpha=0.3, linestyle="--", linewidth=0.5)

    axes[1].hist(label_np, bins=bins, color="tab:orange", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label distribution")
    axes[1].grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved label stream visualization to {output_path}")


if __name__ == "__main__":
    parse_args()
