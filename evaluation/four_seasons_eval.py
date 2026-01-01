from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate
import matplotlib

# non-interactive backend for CLI/server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collate_first_two(batch):
    """Drop any extra fields; keep only (x, y)."""
    pairs = [(item[0], item[1]) for item in batch]
    return default_collate(pairs)  # type: ignore


def resolve_stats_path(base_dir: str | Path, dataset: str, scene: str, filename: str) -> Path:
    base_dir = Path(base_dir)
    # Prefer path without repeating dataset folder if base_dir already points to it.
    candidates = [
        base_dir / scene / filename,
        base_dir / dataset / scene / filename,
    ]
    if str(dataset).lower() == "4seasons":
        candidates.append(base_dir / f"{dataset}_poses" / scene / filename)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def qexp(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return np.hstack((np.cos(n), np.sinc(n / np.pi) * q))


def quaternion_angular_error(q1: np.ndarray, q2: np.ndarray) -> float:
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    return float(2 * np.arccos(d) * 180 / np.pi)


def plot_four_seasons(
    pred: np.ndarray,
    targ: np.ndarray,
    t_loss: np.ndarray,
    pose_m: np.ndarray,
    pose_s: np.ndarray,
    fig_path: Path,
) -> None:
    """Plot predicted/GT trajectories colored by translation error (similar to code1_v1_4seasons/eval.py)."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    real_pose = (pred[:, :3] - pose_m) / pose_s
    gt_pose = (targ[:, :3] - pose_m) / pose_s

    fig, _ = plt.subplots()
    plt.plot(real_pose[:, 1], real_pose[:, 0], color="red", linewidth=0.5)

    norm = plt.Normalize(t_loss.min(), t_loss.max()) # type: ignore
    colors = norm(t_loss)
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], c=colors, cmap="jet", linewidths=1)

    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], "y*", markersize=15)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def evaluate_four_seasons(
    regressor: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ds_cfg: dict[str, Any],
    fig_path: str | Path | None = None,
) -> dict[str, float]:
    """Eval translation/rotation errors like code1_v1_4seasons/eval.py."""
    regressor.eval()
    scene = ds_cfg.get("scene", "neighborhood")
    base = ds_cfg.get("data_path", "data/4Seasons")
    dataset_name = ds_cfg.get("dataset", "4Seasons")
    pose_stats_file = resolve_stats_path(base, dataset_name, scene, "pose_stats.txt")
    pose_m, pose_s = np.loadtxt(pose_stats_file)

    pred_list, targ_list = [], []
    device = next(regressor.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.cpu().numpy()

            feat = regressor.feature(x) # type: ignore
            out = regressor.predict_from_feature(feat).cpu().numpy() # type: ignore

            q_out = np.asarray([qexp(p[3:]) for p in out])
            q_targ = np.asarray([qexp(t[3:]) for t in y])
            out_full = np.hstack((out[:, :3], q_out))
            targ_full = np.hstack((y[:, :3], q_targ))

            out_full[:, :3] = (out_full[:, :3] * pose_s) + pose_m
            targ_full[:, :3] = (targ_full[:, :3] * pose_s) + pose_m

            pred_list.append(out_full)
            targ_list.append(targ_full)

    pred = np.vstack(pred_list)
    targ = np.vstack(targ_list)
    t_loss = np.asarray([np.linalg.norm(p - t) for p, t in zip(pred[:, :3], targ[:, :3])])
    q_loss = np.asarray([quaternion_angular_error(p, t) for p, t in zip(pred[:, 3:], targ[:, 3:])])

    if fig_path is not None:
        plot_four_seasons(pred, targ, t_loss, pose_m, pose_s, Path(fig_path))

    return {
        "t_median": float(np.median(t_loss)),
        "q_median": float(np.median(q_loss)),
        "t_mean": float(np.mean(t_loss)),
        "q_mean": float(np.mean(q_loss)),
    }


def evaluate_four_seasons_online(
    regressor: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ds_cfg: dict[str, Any],
    fig_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Streaming-style evaluation (single pass, no extra optimizer steps).
    Can be called after TTA to get per-sample error traces (model fixed).
    """
    regressor.eval()
    scene = ds_cfg.get("scene", "neighborhood")
    base = ds_cfg.get("data_path", "data/4Seasons")
    dataset_name = ds_cfg.get("dataset", "4Seasons")
    pose_stats_file = resolve_stats_path(base, dataset_name, scene, "pose_stats.txt")
    pose_m, pose_s = np.loadtxt(pose_stats_file)

    pred_list, targ_list = [], []
    device = next(regressor.parameters()).device
    t_loss_series: list[float] = []
    q_loss_series: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y_np = y.cpu().numpy()

            feat = regressor.feature(x)  # type: ignore
            out = regressor.predict_from_feature(feat).cpu().numpy()  # type: ignore

            q_out = np.asarray([qexp(p[3:]) for p in out])
            q_targ = np.asarray([qexp(t[3:]) for t in y_np])
            out_full = np.hstack((out[:, :3], q_out))
            targ_full = np.hstack((y_np[:, :3], q_targ))

            out_full[:, :3] = (out_full[:, :3] * pose_s) + pose_m
            targ_full[:, :3] = (targ_full[:, :3] * pose_s) + pose_m

            pred_list.append(out_full)
            targ_list.append(targ_full)

            t_loss_batch = np.linalg.norm(out_full[:, :3] - targ_full[:, :3], axis=1)
            q_loss_batch = np.asarray([quaternion_angular_error(p, t) for p, t in zip(out_full[:, 3:], targ_full[:, 3:])])
            t_loss_series.extend(t_loss_batch.tolist())
            q_loss_series.extend(q_loss_batch.tolist())

    pred = np.vstack(pred_list)
    targ = np.vstack(targ_list)
    t_loss = np.asarray(t_loss_series)
    q_loss = np.asarray(q_loss_series)

    if fig_path is not None:
        plot_four_seasons(pred, targ, t_loss, pose_m, pose_s, Path(fig_path))

    return {
        "t_median": float(np.median(t_loss)),
        "q_median": float(np.median(q_loss)),
        "t_mean": float(np.mean(t_loss)),
        "q_mean": float(np.mean(q_loss)),
    }


class FourSeasonsOnlineTracker:
    """Attach to an Ignite engine to track online errors while the model is adapting."""

    def __init__(self, ds_cfg: dict[str, Any]):
        scene = ds_cfg.get("scene", "neighborhood")
        base = ds_cfg.get("data_path", "data/4Seasons")
        dataset_name = ds_cfg.get("dataset", "4Seasons")
        pose_stats_file = resolve_stats_path(base, dataset_name, scene, "pose_stats.txt")
        self.pose_m, self.pose_s = np.loadtxt(pose_stats_file)

        self.pred_list: list[np.ndarray] = []
        self.targ_list: list[np.ndarray] = []
        self.t_loss_series: list[float] = []
        self.q_loss_series: list[float] = []

    def update(self, engine) -> None:
        out = engine.state.output
        y_pred = out["y_pred"].detach().cpu().numpy()
        y_true = out["y"].detach().cpu().numpy()

        q_out = np.asarray([qexp(p[3:]) for p in y_pred])
        q_targ = np.asarray([qexp(t[3:]) for t in y_true])
        out_full = np.hstack((y_pred[:, :3], q_out))
        targ_full = np.hstack((y_true[:, :3], q_targ))

        out_full[:, :3] = (out_full[:, :3] * self.pose_s) + self.pose_m
        targ_full[:, :3] = (targ_full[:, :3] * self.pose_s) + self.pose_m

        self.pred_list.append(out_full)
        self.targ_list.append(targ_full)

        t_loss_batch = np.linalg.norm(out_full[:, :3] - targ_full[:, :3], axis=1)
        q_loss_batch = np.asarray(
            [quaternion_angular_error(p, t) for p, t in zip(out_full[:, 3:], targ_full[:, 3:])]
        )
        self.t_loss_series.extend(t_loss_batch.tolist())
        self.q_loss_series.extend(q_loss_batch.tolist())

    def compute(self, fig_path: str | Path | None = None) -> dict[str, Any]:
        pred = np.vstack(self.pred_list) if self.pred_list else np.empty((0, 7))
        targ = np.vstack(self.targ_list) if self.targ_list else np.empty((0, 7))
        t_loss = np.asarray(self.t_loss_series)
        q_loss = np.asarray(self.q_loss_series)

        if fig_path is not None and pred.size > 0:
            plot_four_seasons(pred, targ, t_loss, self.pose_m, self.pose_s, Path(fig_path))

        return {
            "t_median": float(np.median(t_loss)) if t_loss.size else float("nan"),
            "q_median": float(np.median(q_loss)) if q_loss.size else float("nan"),
            "t_mean": float(np.mean(t_loss)) if t_loss.size else float("nan"),
            "q_mean": float(np.mean(q_loss)) if q_loss.size else float("nan"),
        }
