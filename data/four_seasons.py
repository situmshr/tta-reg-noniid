"""4Seasons neighborhood-only dataloader with alias support and simple split.

This module intentionally focuses on the single scene ``neighborhood``.
The dataset uses *only* the sequences (aliases or recording names) provided via config.

Assumption:
- ``subseq_length`` is fixed to 1 (single-frame samples). Related "subsequence" logic
  (e.g., ``skip`` offsets) is intentionally omitted for simplicity.
"""
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import transforms3d.quaternions as txq
import torchvision.transforms.functional as TVF
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# Alias -> recording name
SEQ_ALIASES = {
    "neighborhood_1": "recording_2020-03-26_13-32-55",
    "neighborhood_2": "recording_2020-10-07_14-47-51",
    "neighborhood_3": "recording_2020-10-07_14-53-52",
    "neighborhood_4": "recording_2020-12-22_11-54-24",
    "neighborhood_5": "recording_2021-02-25_13-25-15",
    "neighborhood_6": "recording_2021-05-10_18-02-12",
    "neighborhood_7": "recording_2021-05-10_18-32-32",
}


def resolve_seq_ids(seq_ids: Iterable[str]) -> list[str]:
    return [SEQ_ALIASES.get(s, s) for s in seq_ids]


def qlog(q):
    if all(q[1:] == 0):
        return np.zeros(3)
    return np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def split_continuous_trajectory(
    items: list[int], chunk_size: int = 16, test_ratio: float = 0.125, seed: int = 7
) -> tuple[list[int], list[int]]:
    """
    Split list of items (already time-ordered) into train/test by contiguous chunks.
    """
    random.seed(seed)
    if len(items) == 0:
        return [], []
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    total_chunks = len(chunks)
    num_test_chunks = int(total_chunks * test_ratio)
    test_chunk_indices = (
        set(random.sample(range(total_chunks), num_test_chunks)) if num_test_chunks > 0 else set()
    )
    train_items, test_items = [], []
    for idx, chunk in enumerate(chunks):
        (test_items if idx in test_chunk_indices else train_items).extend(chunk)
    return train_items, test_items


def resolve_roots(data_path: str | Path, scene: str) -> tuple[Path, Path]:
    """
    Resolve image/pose roots from ``data_path``.

    Supported layouts:
    - data_path == "data/4Seasons" (expects "data/4Seasons/<scene>/...")
    - data_path == "data"          (expects "data/4Seasons/<scene>/...")
    Pose root is expected under the same 4Seasons tree:
      - <img_base>/<scene>/*.txt (e.g., pose_stats.txt, <recording>_tR.txt)
    """
    data_path = Path(data_path)

    # images
    if (data_path / scene).is_dir():
        img_base = data_path
    elif (data_path / "4Seasons" / scene).is_dir():
        img_base = data_path / "4Seasons"
    else:
        raise FileNotFoundError(
            f"4Seasons image root not found under {data_path!r}; expected "
            f"{data_path}/neighborhood or {data_path}/4Seasons/neighborhood."
        )

    # poses: colocated under 4Seasons/<scene>
    pose_base = img_base
    pose_dir = pose_base / scene
    has_pose_files = (pose_dir / "pose_stats.txt").exists() or any(pose_dir.glob("*_tR.txt"))
    if not has_pose_files:
        raise FileNotFoundError(
            f"4Seasons pose files not found under {pose_dir}; expected pose_stats.txt and/or '*_tR.txt'."
        )
    return img_base, pose_base


def resolve_stats_path(img_base: Path, pose_base: Path, scene: str, filename: str) -> Path:
    for cand in (img_base / scene / filename, pose_base / scene / filename):
        if cand.exists():
            return cand
    return img_base / scene / filename


class FourSeasonsDataset(data.Dataset):
    """Neighborhood-only loader. `seqs` is a list of alias keys (neighborhood_1...)."""

    def __init__(
        self,
        data_path: str,
        seqs: list[str],
        train: bool,
        transform=None,
        target_transform=None,
        seed: int = 7,
        random_crop: bool = True,
        cropsize: int = 128,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.random_crop = random_crop
        self.cropsize = cropsize

        seqs_resolved = resolve_seq_ids(seqs)
        if not seqs_resolved:
            raise ValueError("No sequences provided for FourSeasonsDataset")

        scene = "neighborhood"
        img_base, pose_base = resolve_roots(data_path, scene)
        img_root = img_base / scene
        pose_root = pose_base / scene

        self.img_paths = []
        self.poses = np.empty((0, 6))

        pose_stats_file = resolve_stats_path(img_base, pose_base, scene, "pose_stats.txt")
        mean_t, std_t = np.loadtxt(pose_stats_file)

        for seq in seqs_resolved:
            img_dir = img_root / seq / f"cam0_{cropsize}"
            if not img_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            timestamps = sorted(
                int(p.stem) for p in img_dir.iterdir() if p.is_file() and p.suffix == ".png"
            )
            img_list = [img_dir / f"{t}.png" for t in timestamps]

            pose_file = pose_root / f"{seq}_tR.txt"
            poses_np = process_poses(
                np.loadtxt(pose_file), mean_t, std_t, np.eye(3), np.zeros(3), 1
            )

            if len(img_list) != len(poses_np):
                raise ValueError(f"Image/pose length mismatch for {seq}: {len(img_list)} vs {len(poses_np)}")

            self.img_paths.extend(img_list)
            self.poses = np.vstack((self.poses, poses_np))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        timestamp = torch.tensor(int(img_path.stem))

        img = default_loader(str(img_path))
        if self.random_crop:
            i, j, th, tw = transforms.RandomCrop(size=self.cropsize).get_params(
                img, output_size=[self.cropsize, self.cropsize] # type: ignore
            )
            img = TVF.crop(img, i, j, th, tw)
        else:
            img = transforms.CenterCrop(self.cropsize)(img)

        if self.transform is not None:
            img = self.transform(img)

        pose = np.float32(self.poses[index])
        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.train:
            return img, pose, timestamp
        else:
            return img, pose, img_path.stem, timestamp


def get_four_seasons(config: dict) -> Tuple[data.Dataset, data.Dataset]:
    ds_cfg = config["dataset"]["config"]
    data_path = ds_cfg["data_path"]
    scene = ds_cfg.get("scene", "neighborhood")
    if scene != "neighborhood":
        raise ValueError("data.four_seasons currently supports only scene='neighborhood'")

    # Use only sequences explicitly listed in the config.
    seqs = list(ds_cfg["seqs"])
    train_ratio = config["dataset"].get("train_ratio", 1.0)
    train_flag = ds_cfg.get("train", True)

    split_cfg = ds_cfg.get("contiguous_split")
    use_contiguous_split = False
    split_chunk_size = 16
    split_seed = config.get("seed", 7)
    if isinstance(split_cfg, bool):
        use_contiguous_split = split_cfg
    elif isinstance(split_cfg, dict):
        use_contiguous_split = split_cfg.get("enabled", True)
        split_chunk_size = split_cfg.get("chunk_size", split_chunk_size)
        split_seed = split_cfg.get("seed", split_seed)

    if ds_cfg.get("subseq_length", 1) != 1:
        raise ValueError("data.four_seasons currently supports only subseq_length=1")

    # transforms
    img_base, pose_base = resolve_roots(data_path, scene)
    stats_file = resolve_stats_path(img_base, pose_base, scene, "stats.txt")
    mean, std = np.loadtxt(stats_file)
    tforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=np.sqrt(std)),
    ])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    common_kwargs = dict(
        data_path=data_path,
        seqs=seqs,
        cropsize=ds_cfg.get("cropsize", 128),
        seed=config.get("seed", 7),
        target_transform=target_transform,
    )


    # full dataset for splitting
    full_ds = FourSeasonsDataset(
        train=train_flag,
        random_crop=ds_cfg.get("random_crop", True),
        transform=tforms,
        **common_kwargs, # type: ignore
    )
    n = len(full_ds)
    if use_contiguous_split:
        indices = list(range(n))
        train_idx, val_idx = split_continuous_trajectory(
            indices,
            chunk_size=split_chunk_size,
            test_ratio=1 - train_ratio,
            seed=split_seed,
        )
        if not train_idx:
            train_ds = data.Subset(full_ds, [])
            val_base = FourSeasonsDataset(
                train=False,
                random_crop=False,
                transform=tforms,
                **common_kwargs, # type: ignore
            )
            val_ds = data.Subset(val_base, val_idx) if val_idx else val_base
        else:
            train_ds = data.Subset(full_ds, train_idx)
            val_ds = data.Subset(full_ds, val_idx) if val_idx else full_ds
    else:
        # No splitting; use the full dataset for both train and validation.
        train_ds = full_ds
        val_ds = full_ds

    return train_ds, val_ds
