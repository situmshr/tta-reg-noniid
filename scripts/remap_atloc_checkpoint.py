#!/usr/bin/env python3
"""
Remap AtLocSimple checkpoints (backbone/head) to the CNNRegressor format
used in models/arch/Res.py (feature_extractor/regressor).

Example:
    python3 scripts/remap_atloc_checkpoint.py \
        --input path/to/epoch_10.pth.tar \
        --output path/to/epoch_10_res_format.pth.tar
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Set

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import timm
from torch.nn.modules.normalization import GroupNorm as TorchGroupNorm
from timm.layers.norm import GroupNorm as TimmGroupNorm


PREFIX_MAP = {
    "backbone.": "feature_extractor.",
    "head.": "regressor.",
}


def resolve_state_dict(container: object) -> Tuple[Dict[str, torch.Tensor], str]:
    """Return (state_dict, key_name) from a checkpoint-like object.

    If the input is not a dict, it is assumed to already be a state_dict and
    key_name will be None.
    """

    if not isinstance(container, dict):
        return container, None

    for key in ("model_state_dict", "state_dict", "model"):
        maybe_state = container.get(key)
        if isinstance(maybe_state, dict):
            return maybe_state, key

    # Fallback: treat the whole container as the state_dict
    return container, None


def build_allowed_keys(backbone: str, pretrained: bool, in_channels: int, out_dim: int) -> Set[str]:
    """Recreate CNNRegressor state_dict keys without importing Res.py (avoids Python 3.10 match)."""

    backbone = backbone.strip()
    if backbone == "resnet26-BN":
        base_net = timm.create_model("resnet26", pretrained=pretrained)
        if in_channels != 3:
            base_net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        feature_extractor = create_feature_extractor(base_net, {"global_pool": "feature"})
    elif backbone == "resnet50-BN":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_net = resnet50(weights=weights)
        if in_channels != 3:
            base_net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        feature_extractor = create_feature_extractor(base_net, {"avgpool": "feature"})
    elif backbone == "resnet50-GN":
        base_net = timm.create_model("resnet50_gn", pretrained=pretrained)
        if in_channels != 3:
            base_net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        feature_extractor = create_feature_extractor(
            base_net,
            {"global_pool": "feature"},
            tracer_kwargs={"leaf_modules": [TorchGroupNorm, TimmGroupNorm]},
        )
    else:
        raise ValueError(f"Invalid backbone: {backbone!r}")

    allowed = {f"feature_extractor.{k}" for k in feature_extractor.state_dict().keys()}
    regressor = nn.Linear(2048, out_dim)
    allowed.update({f"regressor.{k}" for k in regressor.state_dict().keys()})
    return allowed


def remap_keys(
    state_dict: Dict[str, torch.Tensor], allowed_keys: Optional[Iterable[str]]
) -> Tuple[OrderedDict, Dict[str, int]]:
    """Rename prefixes from AtLocSimple to CNNRegressor expected names."""

    allowed = set(allowed_keys) if allowed_keys is not None else None
    remapped = OrderedDict()
    stats = {"remapped": 0, "unchanged": 0, "dropped_unknown": 0, "missing_after": 0}

    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in PREFIX_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                stats["remapped"] += 1
                break
        else:
            stats["unchanged"] += 1

        if allowed is not None and new_key not in allowed:
            stats["dropped_unknown"] += 1
            continue

        remapped[new_key] = value

    if allowed is not None:
        stats["missing_after"] = len(allowed - set(remapped.keys()))

    return remapped, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an AtLocSimple checkpoint (with `backbone`/`head` keys) "
            "so it can be loaded by models/arch/Res.py (expects "
            "`feature_extractor`/`regressor`). Keys not present in the target "
            "architecture will be dropped."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the source checkpoint")
    parser.add_argument("--output", required=True, help="Where to write the converted checkpoint")
    parser.add_argument(
        "--backbone",
        default="resnet50-GN",
        choices=["resnet26-BN", "resnet50-BN", "resnet50-GN"],
        help="CNNRegressor backbone to build for filtering keys",
    )
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels for CNNRegressor")
    parser.add_argument("--out-dim", type=int, default=6, help="Output dimension for CNNRegressor")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to load the backbone with pretrained weights (defaults to False)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show a summary without writing the converted file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict, state_key = resolve_state_dict(checkpoint)

    allowed_keys = build_allowed_keys(
        backbone=args.backbone,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        out_dim=args.out_dim,
    )

    new_state_dict, stats = remap_keys(state_dict, allowed_keys)

    print(
        f"Remapped keys: {stats['remapped']} | Unchanged: {stats['unchanged']} | "
        f"Dropped (unknown): {stats['dropped_unknown']} | Missing after remap: {stats['missing_after']}"
    )
    if args.dry_run:
        return

    if isinstance(checkpoint, dict):
        checkpoint_out = checkpoint.copy()
        target_key = state_key or "model_state_dict"
        checkpoint_out[target_key] = new_state_dict
    else:
        checkpoint_out = new_state_dict

    torch.save(checkpoint_out, args.output)
    print(f"Saved converted checkpoint to {args.output}")


if __name__ == "__main__":
    main()
