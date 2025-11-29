from typing import Any

from torch.utils.data import Dataset

from .utkface import get_utkface
from .non_iid import get_note_non_iid_dataset

__all__ = ["get_datasets", "get_note_non_iid_dataset"]


def get_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    train_aug = config["dataset"].get("train_aug", True)
    print("train_aug:", train_aug)

    match name := config["dataset"]["name"]:
        case "utkface":
            train_ds, val_ds = get_utkface(config)
        case _:
            raise ValueError(f"Invalid dataset: {name!r}")

    print(f"dataset: {name}")

    return train_ds, val_ds