from typing import Any
from collections.abc import Iterator

from torch import nn, Tensor
from torch.nn.modules.normalization import LayerNorm
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import timm


from .Reg import Regressor


class ViTRegressor(Regressor):
    """
    Vision Transformer based regressor mirroring the ResNet regressor interface.
    """

    def __init__(self, backbone: str, pretrained: bool, in_channels: int,
                 global_pool: str = "avg") -> None:
        super().__init__()

        match backbone:
            case "vit-b-16-224-LN":
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                base_net = vit_b_16(weights=weights)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"encoder.ln": "feature"})

            case "deit_base_patch16_224":
                base_net = timm.create_model(
                    "deit_base_patch16_224",
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=in_channels,
                    global_pool=global_pool,
                )
                
            case _:
                raise ValueError(f"Invalid backbone: {backbone!r}")

        self.regressor = nn.Linear(768, 1)

    def feature(self, x: Tensor) -> Tensor:
        z: Tensor = self.feature_extractor(x)["feature"]
        return z[:, 0]

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Module:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


def extract_ln_layers(mod: nn.Module) -> Iterator[LayerNorm]:
    for m in mod.children():
        if isinstance(m, LayerNorm):
            yield m
        else:
            yield from extract_ln_layers(m)
