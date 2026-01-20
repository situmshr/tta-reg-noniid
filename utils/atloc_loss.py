import torch
from torch import nn

__all__ = [
    "calc_vos_simple_batch",
    "AtLocCriterion",
    "AtLocPlusCriterion",
]


def calc_vos_simple_batch(poses: torch.Tensor) -> torch.Tensor:
    if poses.dim() == 3:
        return poses[:, 1:, :] - poses[:, :-1, :]
    return poses[1:, :] - poses[:-1, :]


class AtLocCriterion(nn.Module):
    def __init__(
        self,
        t_loss_fn: nn.Module = nn.L1Loss(),
        q_loss_fn: nn.Module = nn.L1Loss(),
        sax: float = 0.0,
        saq: float = 0.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        abs_loss = (
            torch.exp(-self.sax) * self.t_loss_fn(pred[..., :3], targ[..., :3])
            + self.sax
            + torch.exp(-self.saq) * self.q_loss_fn(pred[..., 3:], targ[..., 3:])
            + self.saq
        )
        return abs_loss


class AtLocPlusCriterion(nn.Module):
    def __init__(
        self,
        t_loss_fn: nn.Module = nn.L1Loss(),
        q_loss_fn: nn.Module = nn.L1Loss(),
        sax: float = 0.0,
        saq: float = 0.0,
        srx: float = 0.0,
        srq: float = 0.0,
        learn_beta: bool = True,
        learn_gamma: bool = True,
    ):
        super().__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        abs_loss = (
            torch.exp(-self.sax) * self.t_loss_fn(pred[..., :3], targ[..., :3])
            + self.sax
            + torch.exp(-self.saq) * self.q_loss_fn(pred[..., 3:], targ[..., 3:])
            + self.saq
        )

        pred_vos = calc_vos_simple_batch(pred)
        targ_vos = calc_vos_simple_batch(targ)
        if pred_vos.numel() == 0:
            vo_loss = pred_vos.sum() * 0.0
        else:
            vo_loss = (
                torch.exp(-self.srx)
                * self.t_loss_fn(pred_vos[..., :3], targ_vos[..., :3])
                + self.srx
            )

        loss = abs_loss + vo_loss
        return loss
