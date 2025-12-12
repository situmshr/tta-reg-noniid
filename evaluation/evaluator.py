from dataclasses import dataclass, InitVar
from typing import Any, Sequence

import torch
from torch import Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError, Average
from ignite.contrib.metrics.regression.r2_score import R2Score

from models.arch import Regressor
from .pca_basis import get_pca_basis
from .metrics import ModelDistanceMetric, PearsonCorrelation, FeatureKLMetric


@dataclass
class RegressionEvaluator(Engine):
    net: Regressor
    pc_config: InitVar[dict | None] = None
    raw_feat_pt: InitVar[str | None] = None
    compile_model: InitVar[dict | None] = None
    val_dataset: InitVar[Any | None] = None
    target_names: InitVar[Sequence[str] | None] = None

    def __post_init__(self,
                      pc_config: dict | None,
                      raw_feat_pt: str | None,
                      compile_model: dict | None,
                      val_dataset: Any | None,
                      target_names: Sequence[str] | None):
        super().__init__(self.inference)

        names_from_ds = self._extract_target_names_from_dataset(val_dataset)
        self._target_names = self._prepare_target_names(
            target_names, getattr(self.net, "out_dim", None), names_from_ds
        )

        RootMeanSquaredError(self._overall_output).attach(self, "rmse_loss")
        MeanAbsoluteError(self._overall_output).attach(self, "mae_loss")
        R2Score(self._overall_output).attach(self, "R2")
        PearsonCorrelation(self._overall_output).attach(self, "r")

        for idx, name in enumerate(self._target_names):
            ot = self._per_target_output(idx)
            MeanAbsoluteError(ot).attach(self, f"mae_{name}")
            R2Score(ot).attach(self, f"R2_{name}")

        ModelDistanceMetric(self.net).attach(self, "model_dist")

        if pc_config is not None:
            self.mean, self.pc_basis, self.pc_var = get_pca_basis(**pc_config)

            self.mean = self.mean.cuda()
            self.pc_basis = self.pc_basis.cuda()
            self.pc_var = self.pc_var.cuda()

            if raw_feat_pt is not None:
                raw_features = torch.load(raw_feat_pt)
                self.raw_var = torch.var(raw_features, dim=0).cuda()
            else:
                raise ValueError("raw_feat_pt must be provided when pc_config is set")

            pc_ot = lambda d: d["feat_pc"]
            zeros = torch.zeros_like(self.pc_var)
            FeatureKLMetric(pc_ot, zeros, self.pc_var, reverse_kl=False) \
                .attach(self, "pc_kl")
            FeatureKLMetric(pc_ot, zeros, self.pc_var, reverse_kl=True) \
                .attach(self, "rev_pc_kl")

            raw_ot = lambda d: d["feat_raw"]
            FeatureKLMetric(raw_ot, self.mean, self.raw_var, reverse_kl=False) \
                .attach(self, "raw_kl")
            FeatureKLMetric(raw_ot, self.mean, self.raw_var, reverse_kl=True) \
                .attach(self, "rev_raw_kl")
        else:
            self.mean = None

        if compile_model is None:
            self.feature_extractor = self.net.feature
        else:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.feature_extractor = self.net.feature

    @torch.no_grad()
    def inference(self,
                  engine: Engine,
                  batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.eval()

        x, y = batch
        x = x.cuda()
        y = y.float().cuda()

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        output = {
            "y_pred": y_pred,
            "y": y,
            "feat_raw": feature
        }
        if self.mean is not None:
            output["feat_pc"] = (feature - self.mean) @ self.pc_basis
        return output

    def _prepare_target_names(
        self, provided: Sequence[str] | None, out_dim: int | None, from_dataset: Sequence[str]
    ) -> list[str]:
        if from_dataset:
            return list(from_dataset)
        if provided is not None:
            return list(provided)
        if isinstance(out_dim, int) and out_dim > 1:
            if out_dim == 3:
                return ["yaw", "roll", "pitch"]
            return [f"target_{i}" for i in range(out_dim)]
        return []

    def _extract_target_names_from_dataset(self, dataset: Any | None) -> list[str]:
        """Traverse wrapped datasets (Subset/ImageTransformDataset, etc.) to find a `target` attribute."""
        visited: set[int] = set()

        def _recurse(obj: Any | None) -> list[str]:
            if obj is None:
                return []
            obj_id = id(obj)
            if obj_id in visited:
                return []
            visited.add(obj_id)

            tgt = getattr(obj, "target", None)
            if isinstance(tgt, (list, tuple)) and tgt:
                return [str(x) for x in tgt]
            if isinstance(tgt, str) and tgt:
                return [tgt]

            for attr in ("dataset", "datasets", "data"):
                child = getattr(obj, attr, None)
                if child is None:
                    continue
                if isinstance(child, (list, tuple)):
                    for c in child:
                        res = _recurse(c)
                        if res:
                            return res
                else:
                    res = _recurse(child)
                    if res:
                        return res
            return []

        return _recurse(dataset)

    def _reshape_outputs(self, output: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        y_true = output["y"]
        y_pred = output["y_pred"]

        if y_pred.ndim == 1 and y_true.ndim > 1:
            y_pred = y_pred.view(y_true.shape[0], -1)
        elif y_pred.ndim > 1 and y_true.ndim == 1:
            y_true = y_true.view_as(y_pred)

        return y_pred, y_true

    def _overall_output(self, output: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        y_pred, y_true = self._reshape_outputs(output)
        return y_pred.reshape(-1), y_true.reshape(-1)

    def _per_target_output(self, idx: int):
        def _ot(output: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
            y_pred, y_true = self._reshape_outputs(output)
            if y_pred.ndim == 1:
                return y_pred, y_true
            return y_pred[:, idx], y_true[:, idx]

        return _ot
    

@dataclass
class VBLLEvaluator(Engine):
    vbll_head: torch.nn.Module

    def __post_init__(self):
        super().__init__(self.inference)

        RootMeanSquaredError(
            output_transform=lambda out: (out["y_pred"], out["y"])
        ).attach(self, "rmse")
        MeanAbsoluteError(
            output_transform=lambda out: (out["y_pred"], out["y"])
        ).attach(self, "mae")
        R2Score(
            output_transform=lambda out: (out["y_pred"], out["y"])
        ).attach(self, "R2")
        PearsonCorrelation(
            output_transform=lambda out: (out["y_pred"], out["y"])
        ).attach(self, "r")

        Average(
            output_transform=lambda out: out["nll"]
        ).attach(self, "nll")
        Average(
            output_transform=lambda out: out["y_var"]
        ).attach(self, "var")

    @torch.no_grad()
    def inference(self,
                  engine: Engine,
                  batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.vbll_head.eval()

        x, y = batch
        x = x.cuda()
        y = y.float().flatten().cuda()

        vbll_out = self.vbll_head(x)
        preds = vbll_out.predictive.mean.view_as(y)
        vars = vbll_out.predictive.variance.mean()
        nll = -vbll_out.predictive.log_prob(y.view(-1, 1)).mean()

        return {
            "y_pred": preds,
            "y_var": vars,
            "nll": nll,
            "y": y
        }
