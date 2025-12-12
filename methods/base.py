from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, InitVar
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation


@dataclass
class BaseTTA(Engine, ABC):
    """Common test-time adaptation engine template."""

    net: nn.Module | Any
    opt: torch.optim.Optimizer | None
    train_mode: bool = True
    compile_model: InitVar[dict | None] = None
    val_dataset: InitVar[Any | None] = None
    target_names: InitVar[Sequence[str] | None] = None

    def __post_init__(self, compile_model: dict | None, val_dataset: Any | None, target_names: Sequence[str] | None):
        super().__init__(self._update)

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if compile_model is not None:
            try:
                self.net = torch.compile(self.net, **compile_model)
            except RuntimeError as err:  # pragma: no cover - best effort
                print(f"torch.compile for TTA net failed: {err}")

    def _update(self, engine: Engine, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).float()

        loss: Tensor | None = None

        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

        output, loss = self.adapt_step(x, y)

        if loss is not None and self.opt is not None:
            loss.backward()
            self.opt.step()

        output["y"] = y
        return output

    @abstractmethod
    def adapt_step(self, x: Tensor, y: Tensor) -> tuple[dict[str, Tensor], Tensor | None]:
        """Perform a single adaptation step and return (output_dict, loss)."""

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
