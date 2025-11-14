from typing import Any

from .Reg import Regressor

from .Res import CNNRegressor, extract_bn_layers, extract_gn_layers
from .ViT import ViTRegressor
from .VBLL import ModelWithVBLLHead, create_vbll_head

__all__ = [
    "Regressor",
    "create_regressor",
    "CNNRegressor",
    "ViTRegressor",
    "ModelWithVBLLHead",
    "create_vbll_head",
    "extract_bn_layers",
    "extract_gn_layers",
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