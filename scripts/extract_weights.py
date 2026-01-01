"""
Extract weights-only state_dict from a checkpoint and save as a new file.

Example:
    python3 scripts/extract_weights.py \\
        --src code1_v1_4seasons/logs/4Seasons_neighborhood_RobustLoc_16/models/epoch_299.pth.tar \\
        --dst code1_v1_4seasons/logs/4Seasons_neighborhood_RobustLoc_16/models/epoch_299_state.pth
"""
import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract weights-only state_dict from a checkpoint.")
    parser.add_argument("--src", required=True, type=Path, help="Path to the source checkpoint (.pth or .pth.tar).")
    parser.add_argument("--dst", required=True, type=Path, help="Destination path for the extracted state_dict.")
    parser.add_argument(
        "--key",
        default="model_state_dict",
        help="Checkpoint key that holds the model weights (default: model_state_dict).",
    )
    args = parser.parse_args()

    if not args.src.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.src}")

    checkpoint = torch.load(args.src, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is type {type(checkpoint)}, expected a dict with state_dicts.")
    if args.key not in checkpoint:
        available = ", ".join(checkpoint.keys())
        raise KeyError(f"Key '{args.key}' not found in checkpoint. Available keys: {available}")

    state_dict = checkpoint[args.key]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint key '{args.key}' is type {type(state_dict)}, expected dict of tensors.")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.dst)
    print(f"Saved weights-only state_dict to: {args.dst}")


if __name__ == "__main__":
    main()
