from typing import Any
from collections.abc import Iterator

from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.normalization import GroupNorm
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from .Reg import Regressor



class CNNRegressor(Regressor):
    def __init__(self, backbone: str, pretrained: bool, in_channels: int):
        super().__init__()

        match backbone:
            case "resnet26-BN":
                base_net = timm.create_model("resnet26", pretrained=pretrained)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"global_pool": "feature"})

            case "resnet50-BN":
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                base_net = resnet50(weights=weights)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"avgpool": "feature"})

            case "resnet50-GN":
                base_net = timm.create_model('resnet50_gn', pretrained=True)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"global_pool": "feature"})
                
            case _:
                raise ValueError(f"Invalid backbone: {backbone!r}")

        self.regressor = nn.Linear(2048, 1)

    def feature(self, x: Tensor) -> Tensor:
        z: Tensor = self.feature_extractor(x)["feature"]
        return z.flatten(start_dim=1)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Module:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor



def extract_bn_layers(mod: nn.Module) -> Iterator[_BatchNorm]:
    for m in mod.children():
        if isinstance(m, _BatchNorm):
            yield m
        else:
            yield from extract_bn_layers(m)

def extract_gn_layers(mod: nn.Module) -> Iterator[GroupNorm]:
    for m in mod.children():
        if isinstance(m, GroupNorm):
            yield m
        else:
            yield from extract_gn_layers(m)