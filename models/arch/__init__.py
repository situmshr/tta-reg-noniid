from typing import Any

from .Reg import Regressor

from .Res import CNNRegressor
from .ViT import ViTRegressor

__all__ = [
    "Regressor",
    "create_regressor",
    "CNNRegressor",
    "ViTRegressor",
]


def create_regressor(config: dict[str, Any]) -> Regressor:
    
    match config["regressor"]["arch_type"]:
        case "cnn":
            net = CNNRegressor(**config["regressor"]["config"])

        case "vit":
            net = ViTRegressor(**config["regressor"]["config"])

        case _ as t:
            raise ValueError(f"Invalid type: {t!r}")

    return net