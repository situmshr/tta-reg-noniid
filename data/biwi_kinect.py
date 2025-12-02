from pathlib import Path
import json
import tarfile
import io
import re

from torchvision.transforms import v2 as transforms

import pandas as pd
import numpy as np
from PIL import Image

from .data_configs import BIWI_PATH
from .image_utils import (ImageDataset, ImageTransformDataset, ImageSubset,
                          random_split)



class BiwiKinect(ImageDataset):
    # targetをリストでも受け取れるように変更し、sequential引数を追加
    def __init__(self, gender: str, target: str | list[str], sequential: bool = False):
        # targetが文字列の場合はリストに変換して統一的な処理を可能にする
        if isinstance(target, str):
            target = [target]
        for t in target:
            assert t in ("yaw", "pitch", "roll")
        self.target = target

        dir_path = Path(BIWI_PATH)
        self.root_dir: Path | None = None


        if dir_path.exists():
            self.root_dir = dir_path
        else:
            raise FileNotFoundError(
                f"Biwi data not found. Extracted directory at {dir_path}."
            )

        with Path(BIWI_PATH, "gender.json").open("r", encoding="utf-8") as f:
            person_dirs: str = json.load(f)[gender]

        df = {
            "person": [],
            "frame": [],
            "yaw": [],
            "roll": [],
            "pitch": []
        }


        assert self.root_dir is not None
        for person in person_dirs:
            person_dir = self.root_dir / "faces_0" / person
            if not person_dir.exists():
                raise FileNotFoundError(f"Person directory not found: {person_dir}")

            for pose_file in sorted(person_dir.glob("frame_*_pose.txt")):
                with pose_file.open("r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")

                rot_matrix = np.array([
                    [float(x) for x in line.strip().split(" ")]
                    for line in lines[:3]
                ])

                frame = pose_file.name.split("_")[1]

                y, r, p_ = matrix_to_angles(rot_matrix)
                df["person"].append(person)
                df["frame"].append(frame)
                df["yaw"].append(y)
                df["roll"].append(r)
                df["pitch"].append(p_)

        self.metadata = pd.DataFrame(df)

        # sequential=Trueの場合、人物IDとフレーム番号順にデータをソートする
        if sequential:
            self.metadata["frame_int"] = self.metadata["frame"].astype(int)
            self.metadata = self.metadata.sort_values(by=["person", "frame_int"]).reset_index(drop=True)
            del self.metadata["frame_int"]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, i: int) -> tuple[Image.Image, np.ndarray]: # type: ignore
        metadata = self.metadata.iloc[i].to_dict()
        # 指定された複数のターゲットの値をリストで取得し、numpy配列として返す
        y = np.array([metadata[t] for t in self.target], dtype=np.float32)
        # targetが1つの場合でも形状を維持するか、squeezeするかは下流のタスクに依存しますが、ここでは配列として返します

        assert self.root_dir is not None
        img_file = self.root_dir / "faces_0" / metadata["person"] / f"frame_{metadata['frame']}_rgb.png"
        if not img_file.exists():
            raise FileNotFoundError(f"Frame not found: {img_file}")
        img = Image.open(img_file).convert("RGB")

        w, h = img.width, img.height
        c = (w - h) // 2
        img = img.crop((c, 0, w - c, h))

        return img, y


def matrix_to_angles(m: np.ndarray) -> tuple[float, float, float]:
    y = np.arctan2(m[1, 0], m[0, 0])
    r = np.arctan2(-m[2, 0], np.sqrt(m[2, 1] * m[2, 1] + m[2, 2] * m[2, 2]))
    p = np.arctan2(m[2, 1], m[2, 2])
    return y, r, p


class BiwiKinectClassification(BiwiKinect):
    def __init__(self, n_bins: int, gender: str, target: str):
        super().__init__(gender, target)

        self.n_bins = n_bins

    def __getitem__(self, i: int) -> tuple[Image.Image, int]: # type: ignore
        x, y = super().__getitem__(i)

        MAX_RAD = 60 * np.pi / 180
        MIN_RAD = -60 * np.pi / 180

        # 分類タスクの場合は単一ターゲットが前提となることが多いため、最初の要素を取得する形に修正
        if isinstance(y, np.ndarray) and y.size > 1:
            y = y[0] 

        y = np.clip((y - MIN_RAD) * self.n_bins /
                    (MAX_RAD - MIN_RAD), 0, self.n_bins)
        return x, int(y)


def get_biwi_kinect(config: dict, apply_to_tensor: bool = True, classification: bool = False) -> tuple[ImageDataset, ImageDataset]:
    if apply_to_tensor:
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        to_tensor = transforms.ToTensor()

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        to_tensor
    ])

    if config["dataset"].get("train_aug", True):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            to_tensor
        ])
    else:
        train_transform = val_transform

    if classification:
        ds = BiwiKinectClassification(**config["dataset"]["config"])
    else:
        ds = BiwiKinect(**config["dataset"]["config"])

    if "val_indices" in config["dataset"]:
        val_indices: np.ndarray = np.load(config["dataset"]["val_indices"])
        print(
            f"BiwiKinect: load val indices from {config['dataset']['val_indices']}")

        train_mask = np.ones(len(ds), dtype=np.bool_)
        train_mask[val_indices] = False
        train_indices = np.arange(len(ds))[train_mask]

        train_ds = ImageSubset(ds, train_indices.tolist())
        val_ds = ImageSubset(ds, val_indices.tolist())
    else:
        print("BiwiKinect: split randomly")
        n = int(len(ds) * config["dataset"]["train_ratio"])
        train_ds, val_ds = random_split(ds, n)

    train_ds = ImageTransformDataset(train_ds, train_transform)
    val_ds = ImageTransformDataset(val_ds, val_transform)
    return train_ds, val_ds
