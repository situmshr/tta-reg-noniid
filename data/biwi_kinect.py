from pathlib import Path
import json
import re

from torchvision.transforms import v2 as transforms

import pandas as pd
import numpy as np
from PIL import Image

from .data_configs import BIWI_PATH
from .image_utils import (ImageDataset, ImageTransformDataset, ImageSubset,
                          random_split)


class BiwiKinect(ImageDataset):
    def __init__(self, gender: str, target: str | list[str], sequential: bool = False):
        # targetが文字列の場合はリストに変換して統一的な処理を可能にする
        if isinstance(target, str):
            target = [target]
        for t in target:
            assert t in ("yaw", "pitch", "roll")
        self.target = target

        self.root_dir = Path(BIWI_PATH)
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Biwi data not found. Extracted directory at {self.root_dir}."
            )

        with Path(self.root_dir, "gender.json").open("r", encoding="utf-8") as f:
            person_dirs: str = json.load(f)[gender]

        df = {
            "person": [],
            "frame": [],
            "yaw": [],
            "roll": [],
            "pitch": []
        }

        pose_files = [
            str(p.relative_to(self.root_dir))
            for p in self.root_dir.glob("faces_0/*/frame_*_pose.txt")
            if p.is_file()
        ]
        for person in person_dirs:
            metadata_paths = (
                m
                for m in pose_files
                if re.search(rf"faces_0/{person}/frame_\d+_pose.txt", m)
            )

            for p in metadata_paths:
                with Path(self.root_dir, p).open("r", encoding="utf-8") as fp:
                    lines = fp.read().strip().split("\n")

                rot_matrix = np.array([
                    [float(x) for x in l.strip().split(" ")]
                    for l in lines[:3]
                ])

                frame = p.split("/")[-1].split("_")[1]

                y, r, p = matrix_to_angles(rot_matrix)
                df["person"].append(person)
                df["frame"].append(frame)
                df["yaw"].append(y)
                df["roll"].append(r)
                df["pitch"].append(p)

        self.metadata = pd.DataFrame(df)

        # sequential=Trueの場合、人物IDとフレーム番号順にデータをソートする
        if sequential:
            # 文字列のframe番号を数値として正しくソートするため、一時的なint列を作成
            self.metadata["frame_int"] = self.metadata["frame"].astype(int)
            # 人物ID -> フレーム番号の順に並べ替え、データの順序を一意に固定する
            self.metadata = self.metadata.sort_values(by=["person", "frame_int"]).reset_index(drop=True)
            # ソート用に使用した一時カラムを削除
            del self.metadata["frame_int"]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, i: int) -> tuple[Image.Image, np.ndarray | float]:
        metadata = self.metadata.iloc[i].to_dict()
        # 指定された複数のターゲットの値をリストで取得し、numpy配列として返す
        y = np.array([metadata[t] for t in self.target], dtype=np.float32)
        # targetが1つの場合でも形状を維持するか、squeezeするかは下流のタスクに依存しますが、ここでは配列として返します

        img_path = self.root_dir / \
            "faces_0" / metadata["person"] / f"frame_{metadata['frame']}_rgb.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Frame not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        w, h = img.width, img.height
        c = (w - h) // 2
        img = img.crop((c, 0, w - c, h))

        # targetが1つの場合はfloatを返す（単一回帰タスク用）
        if len(self.target) == 1:
            return img, float(y[0])

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

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
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
