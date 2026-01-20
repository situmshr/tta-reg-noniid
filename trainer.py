from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
import torch.nn.functional as F

from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError, Average
from ignite.contrib.metrics.regression.r2_score import R2Score

from models.arch import Regressor
from evaluation.metrics import PearsonCorrelation, OptimizerLastState


@dataclass
class RegressionTrainer(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    compile_model: InitVar[dict | None]
    loss_fn: torch.nn.Module | None = None

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.update)

        y_ot = lambda d: (d["y_pred"].reshape(-1), d["y"].reshape(-1))
        RootMeanSquaredError(y_ot).attach(self, "rmse")
        MeanAbsoluteError(y_ot).attach(self, "mae")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        OptimizerLastState(self.opt, "lr").attach(self, "lr")

        if compile_model is not None:
            try:
                self.net = torch.compile(  # type: ignore
                    self.net, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self,
               engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.train()
        self.opt.zero_grad()

        # OLD:
        # x, y = batch
        # x = x.cuda()
        # y = y.float().cuda()
        #
        # y_pred = self.net(x)
        #
        # loss = F.mse_loss(y_pred, y)

        x = batch[0]
        y = batch[1]
        x = x.cuda()
        y = y.float().cuda()

        b = None
        l = None
        if x.dim() == 5:
            b, l, c, h, w = x.shape
            x = x.view(b * l, c, h, w)

        y_pred = self.net(x)

        y_pred_flat = y_pred
        y_flat = y
        if b is not None and l is not None and y.dim() >= 2:
            y_pred_seq = y_pred.view(b, l, -1)
            y_seq = y.view(b, l, -1)
            if self.loss_fn is not None:
                loss = self.loss_fn(y_pred_seq, y_seq)
            else:
                loss = F.mse_loss(y_pred_seq, y_seq)
            y_pred_flat = y_pred_seq.reshape(-1, y_pred_seq.shape[-1])
            y_flat = y_seq.reshape(-1, y_seq.shape[-1])
        else:
            if self.loss_fn is not None:
                loss = self.loss_fn(y_pred, y)
            else:
                loss = F.mse_loss(y_pred, y)

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred_flat,
            "y": y_flat
        }
    
@dataclass
class VBLLTrainer(Engine):
    vbll_head: torch.nn.Module
    opt: torch.optim.Optimizer

    def __post_init__(self):
        super().__init__(self.update)

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
        
        OptimizerLastState(self.opt, "lr").attach(self, "lr")


    def update(self,
               engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.vbll_head.train()
        self.opt.zero_grad()

        x, y = batch
        x = x.cuda()
        y = y.float().flatten().cuda()

        vbll_out = self.vbll_head(x)
        loss = vbll_out.train_loss_fn(y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vbll_head.parameters(), max_norm=1.0)
        self.opt.step()

        with torch.no_grad():
            y_pred = vbll_out.predictive.mean.view_as(y)
            y_var = vbll_out.predictive.variance.mean()
            nll = -vbll_out.predictive.log_prob(y.view(-1, 1)).mean()

        return {
            "y_pred": y_pred,
            "y_var": y_var,
            "nll": nll,
            "y": y
        }
