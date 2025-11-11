from torch import nn, Tensor


class Regressor(nn.Module):
    regressor: nn.Linear

    def feature(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_from_feature(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def get_regressor(self) -> nn.Module:
        raise NotImplementedError

    def get_feature_extractor(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        z = self.feature(x)
        y_pred = self.predict_from_feature(z)
        return y_pred


