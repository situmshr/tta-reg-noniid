"""Dataset loader for the Biwi Kinect head pose dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import re
import tarfile

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 as transforms

from .dataset_config import BIWI_PATH, PACKAGE_ROOT
from .image_utils import ImageDataset, ImageSubset, ImageTransformDataset, random_split


_TAR_ROOT = "biwi-kinect-head-pose-database"


def _load_metadata(tar: tarfile.TarFile, gender_filter: list[str]) -> pd.DataFrame:
    records = {
        "person": [],
        "frame": [],
        "yaw": [],
        "roll": [],
        "pitch": [],
    }

    members = [m.name for m in tar.getmembers() if m.isfile()]
    for person in gender_filter:
        pose_paths = (
            m
            for m in members
            if re.search(rf"faces_0/{person}/frame_\\d+_pose.txt", m)
        )
        for pose_path in pose_paths:
            file_obj = tar.extractfile(pose_path)
            if file_obj is None:
                continue
            with file_obj:
                lines = file_obj.read().decode("utf-8").strip().split("\n")

            rot_matrix = np.array([[float(x) for x in line.split()] for line in lines[:3]])
            yaw, roll, pitch = _matrix_to_angles(rot_matrix)

            frame = pose_path.split("/")[-1].split("_")[1]
            records["person"].append(person)
            records["frame"].append(frame)
            records["yaw"].append(yaw)
            records["roll"].append(roll)
            records["pitch"].append(pitch)

    return pd.DataFrame(records)


def _matrix_to_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    roll = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
    pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
    return yaw, roll, pitch


@dataclass
class BiwiSample:
    person: str
    frame: str
    target: float


class BiwiKinect(ImageDataset):
    """Regression dataset for Biwi Kinect head pose estimation."""

    def __init__(self, gender: str, target: str):
        if target not in {"yaw", "pitch", "roll"}:
            raise ValueError("target must be one of 'yaw', 'pitch', or 'roll'")

        tar_path = Path(BIWI_PATH, f"{_TAR_ROOT}.tar")
        if not tar_path.exists():
            raise FileNotFoundError(
                f"Biwi archive not found at {tar_path}. Set BIWI_PATH env var to override."
            )

        self.tar = tarfile.open(tar_path)
        self.target = target

        gender_json = Path(BIWI_PATH, "gender.json")
        if not gender_json.exists():
            raise FileNotFoundError(
                f"Gender split file not found at {gender_json}."
            )

        with gender_json.open("r", encoding="utf-8") as f:
            gender_map: dict[str, list[str]] = json.load(f)

        people = gender_map.get(gender)
        if not people:
            raise KeyError(f"Gender '{gender}' not found in gender.json")

        self.metadata = _load_metadata(self.tar, people)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        target_value = float(row[self.target])

        img_path = f"{_TAR_ROOT}/faces_0/{row['person']}/frame_{row['frame']}_rgb.png"
        file_obj = self.tar.extractfile(img_path)
        if file_obj is None:
            raise FileNotFoundError(f"Frame not found inside archive: {img_path}")

        with io.BytesIO(file_obj.read()) as bio:
            image = Image.open(bio).convert("RGB")

        width, height = image.width, image.height
        pad = (width - height) // 2
        if pad > 0:
            image = image.crop((pad, 0, width - pad, height))

        return image, target_value

    def close(self) -> None:
        self.tar.close()


class BiwiKinectClassification(BiwiKinect):
    def __init__(self, n_bins: int, gender: str, target: str):
        super().__init__(gender, target)
        self.n_bins = n_bins

    def __getitem__(self, index: int):
        image, angle = super().__getitem__(index)
        max_rad = np.deg2rad(60)
        min_rad = -max_rad
        bin_idx = np.clip((angle - min_rad) * self.n_bins / (max_rad - min_rad), 0, self.n_bins - 1)
        return image, int(bin_idx)


def get_biwi_kinect(config: dict, apply_to_tensor: bool = True, classification: bool = False):
    if apply_to_tensor:
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        to_tensor = transforms.ToTensor()

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        to_tensor,
    ])

    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("train_aug", True):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            to_tensor,
        ])
    else:
        train_transform = val_transform

    dataset_cls = BiwiKinectClassification if classification else BiwiKinect
    dataset = dataset_cls(**dataset_cfg.get("config", {}))

    if (val_indices_path := dataset_cfg.get("val_indices")) is not None:
        path = Path(val_indices_path)
        if not path.is_absolute():
            path = (PACKAGE_ROOT.parent / path).resolve()
        val_indices = np.load(path)
        print(f"BiwiKinect: load val indices from {path}")

        mask = np.ones(len(dataset), dtype=bool)
        mask[val_indices] = False
        train_indices = np.arange(len(dataset))[mask]

        train_ds = ImageSubset(dataset, train_indices.tolist())
        val_ds = ImageSubset(dataset, val_indices.tolist())
    else:
        print("BiwiKinect: split randomly")
        train_count = int(len(dataset) * dataset_cfg.get("train_ratio", 0.8))
        train_ds, val_ds = random_split(dataset, train_count)

    train_ds = ImageTransformDataset(train_ds, train_transform)
    val_ds = ImageTransformDataset(val_ds, val_transform)

    return train_ds, val_ds
