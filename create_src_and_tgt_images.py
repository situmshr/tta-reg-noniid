#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable

import imagenet_c
import numpy as np
from PIL import Image

from data.data_configs import UTKFACE_PATH


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def apply_corruption(img: Image.Image, corruption_name: str, severity: int) -> Image.Image:
    arr = np.asarray(img)
    corrupted = imagenet_c.corrupt(arr, corruption_name=corruption_name, severity=severity)
    return Image.fromarray(corrupted)


def resolve_ext(name: str | None, fallback: str) -> str:
    if name is None or name == "":
        return fallback
    return name if name.startswith(".") else f".{name}"


def ensure_nonempty(paths: Iterable[Path]) -> list[Path]:
    items = list(paths)
    if not items:
        raise FileNotFoundError("No images found in dataset directory.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample UTKFace images and save originals + ImageNet-C corrupted versions."
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dataset-dir", default=UTKFACE_PATH, help="UTKFace root directory.")
    parser.add_argument("-n", "--num-images", type=int, default=10, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling/corruption.")
    parser.add_argument(
        "--corruption-name",
        default="gaussian_noise",
        help="ImageNet-C corruption name (e.g., gaussian_noise, shot_noise).",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=5,
        help="ImageNet-C severity level (1-5).",
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        help="Output file extension (jpg/png). Default: jpg.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("num-images must be positive.")
    if args.severity <= 0:
        raise ValueError("severity must be positive.")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    image_paths = ensure_nonempty(dataset_dir.glob("*.jpg"))
    if args.num_images > len(image_paths):
        raise ValueError(f"num-images ({args.num_images}) exceeds dataset size ({len(image_paths)}).")

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    selected = rng.choice(image_paths, size=args.num_images, replace=False)

    output_dir = Path(args.output_dir)
    src_dir = output_dir / "source"
    noisy_dir = output_dir / "noisy"
    src_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    ext = resolve_ext(args.ext, ".jpg")
    records = []
    for idx, path in enumerate(selected):
        img = load_image(Path(path))
        noisy = apply_corruption(img, args.corruption_name, args.severity)

        stem = path.stem
        src_name = f"{idx:03d}_{stem}{ext}"
        noisy_name = f"{idx:03d}_{stem}_{args.corruption_name}_s{args.severity}{ext}"

        src_path = src_dir / src_name
        noisy_path = noisy_dir / noisy_name
        img.save(src_path)
        noisy.save(noisy_path)

        records.append(
            {
                "index": idx,
                "source_path": str(src_path),
                "noisy_path": str(noisy_path),
                "original_file": str(path),
                "corruption_name": args.corruption_name,
                "severity": int(args.severity),
            }
        )

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} pairs to {output_dir}")


if __name__ == "__main__":
    main()
