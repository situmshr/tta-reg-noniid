from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, InitVar
from typing import Any

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

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self._update)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

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
        y = y.to(self.device).float().flatten()

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
