#!/usr/bin/env python3
"""Plot 4Seasons neighborhood label trajectories over time.

This script reads pose label files under ``data/4Seasons/neighborhood`` and
plots each label dimension against sample index (ordered by image timestamps).
The x-axis label is kept as "time" for convenience/consistency.

Examples:
  # Plot dataset labels (translation normalized by pose_stats + rotation log-vector)
  ./.venv/bin/python plot_4seasons_labels.py -o outputs/plots/4seasons_labels

  # Plot only one sequence
  ./.venv/bin/python plot_4seasons_labels.py -o outputs/plots/4seasons_labels --seq recording_2020-03-26_13-32-55

  # Plot raw t+q labels
  ./.venv/bin/python plot_4seasons_labels.py -o outputs/plots/4seasons_labels --mode tq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import transforms3d.quaternions as txq
import transforms3d.euler as txe

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Run with the repo venv, e.g. `./.venv/bin/python ...`."
    ) from exc


def qlog(q: np.ndarray) -> np.ndarray:
    if np.all(q[1:] == 0):
        return np.zeros(3, dtype=np.float64)
    return np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])


def qexp(v: np.ndarray) -> np.ndarray:
    """Exponential map from 3D log-quaternion vector to a unit quaternion [w, x, y, z]."""
    n = np.linalg.norm(v)
    return np.hstack((np.cos(n), np.sinc(n / np.pi) * v))


def process_poses_tR(poses_in: np.ndarray, mean_t: np.ndarray, std_t: np.ndarray) -> np.ndarray:
    """Match the 4Seasons dataloader label processing (translation normalized + quaternion log)."""
    poses_out = np.zeros((len(poses_in), 6), dtype=np.float64)
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(R)
        q *= np.sign(q[0])  # constrain to hemisphere
        poses_out[i, 3:] = qlog(q)
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def rvec_to_yaw_pitch_roll_deg(rvec: np.ndarray) -> np.ndarray:
    """Convert qlog 3-vector (SO(3) log / axis-angle/2) into yaw/pitch/roll [deg]."""
    if rvec.ndim != 2 or rvec.shape[1] != 3:
        raise ValueError(f"Expected rvec shape (N, 3), got {rvec.shape}")
    quats = np.asarray([qexp(v) for v in rvec], dtype=np.float64)  # (N, 4) [w,x,y,z]
    # Use static Z-Y-X convention: yaw (Z), pitch (Y), roll (X)
    ypr = np.asarray([txe.quat2euler(q, axes="szyx") for q in quats], dtype=np.float64)  # (N, 3) [yaw,pitch,roll] in rad
    return np.rad2deg(ypr)


def list_sequences(scene_dir: Path) -> list[str]:
    seqs = [
        p.name
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith("recording_")
    ]
    return sorted(seqs)


def load_timestamps(scene_dir: Path, seq: str, cropsize: int) -> np.ndarray:
    img_dir = scene_dir / seq / f"cam0_{cropsize}"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    stamps = sorted(int(p.stem) for p in img_dir.glob("*.png"))
    if not stamps:
        raise FileNotFoundError(f"No PNG images found under {img_dir}")
    return np.asarray(stamps, dtype=np.int64)


def load_labels(scene_dir: Path, seq: str, mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "processed":
        pose_stats = scene_dir / "pose_stats.txt"
        if not pose_stats.is_file():
            raise FileNotFoundError(f"pose_stats.txt not found: {pose_stats}")
        mean_t, std_t = np.loadtxt(pose_stats)

        pose_file = scene_dir / f"{seq}_tR.txt"
        if not pose_file.is_file():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        raw = np.loadtxt(pose_file)
        labels_raw = process_poses_tR(raw, mean_t, std_t)
        ypr_deg = rvec_to_yaw_pitch_roll_deg(labels_raw[:, 3:])
        labels = np.hstack((labels_raw[:, :3], ypr_deg))
        names = [
            "t_x (normalized)",
            "t_y (normalized)",
            "t_z (normalized)",
            "yaw (deg)",
            "pitch (deg)",
            "roll (deg)",
        ]
        return labels, names

    if mode == "tq":
        pose_file = scene_dir / f"{seq}_tq.txt"
        if not pose_file.is_file():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        labels = np.loadtxt(pose_file)
        names = ["t_x (m)", "t_y (m)", "t_z (m)", "q_w", "q_x", "q_y", "q_z"]
        return labels, names

    if mode == "tR":
        pose_file = scene_dir / f"{seq}_tR.txt"
        if not pose_file.is_file():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        raw = np.loadtxt(pose_file)
        labels = raw[:, [3, 7, 11]]
        names = ["t_x (m)", "t_y (m)", "t_z (m)"]
        return labels, names

    raise ValueError(f"Unknown mode: {mode}")


def save_timeseries_plot(
    x_values: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    if y.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {y.shape}")
    if y.shape[1] != len(names):
        raise ValueError(f"Label name mismatch: {y.shape[1]} dims vs {len(names)} names")
    if x_values.shape[0] != y.shape[0]:
        raise ValueError(f"Time/label length mismatch: {x_values.shape[0]} vs {y.shape[0]}")

    fig, axes = plt.subplots(y.shape[1], 1, sharex=True, figsize=(10, 1.8 * y.shape[1]))
    if y.shape[1] == 1:
        axes = [axes]
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.plot(x_values, y[:, i], linewidth=0.8)
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 4Seasons neighborhood labels over time.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data", "4Seasons", "neighborhood"),
        help="Scene directory (default: data/4Seasons/neighborhood).",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Recording name (e.g., recording_2020-03-26_13-32-55). When omitted, plots all.",
    )
    parser.add_argument(
        "--cropsize",
        type=int,
        default=128,
        help="Camera crop size used in folder name cam0_<cropsize> (default: 128).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("processed", "tq", "tR"),
        default="processed",
        help="Which label definition to plot: processed (dataset labels), tq (raw t+quat), tR (raw translation only).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write plot images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir: Path = args.scene_dir
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    seqs = [args.seq] if args.seq is not None else list_sequences(scene_dir)
    if not seqs:
        raise SystemExit(f"No sequences found under {scene_dir}")

    for seq in seqs:
        stamps_ns = load_timestamps(scene_dir, seq, args.cropsize)
        labels, names = load_labels(scene_dir, seq, args.mode)

        if len(labels) != len(stamps_ns):
            raise ValueError(
                f"Length mismatch for {seq}: labels={len(labels)} vs images={len(stamps_ns)} "
                f"(mode={args.mode}, cropsize={args.cropsize})"
            )

        # Plot against sample index (but keep the axis label as "time" for consistency).
        x_values = np.arange(len(labels), dtype=np.float64)
        out_path = args.output_dir / f"{seq}_{args.mode}.png"
        save_timeseries_plot(
            x_values, np.asarray(labels, dtype=np.float64), names, f"{seq} ({args.mode})", out_path
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
