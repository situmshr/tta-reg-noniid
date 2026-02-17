import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_label_stream(dataloader: DataLoader, output_path: Path) -> None:
    labels: list[torch.Tensor] = []
    for _, y in dataloader:
        if isinstance(y, torch.Tensor):
            y_tensor = y.detach().cpu()
        else:
            y_tensor = torch.as_tensor(y).cpu()
        labels.append(y_tensor.float())

    if not labels:
        print("No labels available to visualize.")
        return

    label_tensor = torch.cat(labels, dim=0).float()
    if label_tensor.ndim == 1:
        label_tensor = label_tensor.unsqueeze(1)
    elif label_tensor.ndim > 2:
        label_tensor = label_tensor.view(label_tensor.shape[0], -1)

    num_dims = label_tensor.shape[1]
    idx_np = torch.arange(label_tensor.shape[0]).numpy()
    label_np = label_tensor.numpy()

    if num_dims == 1:
        bins = min(50, max(10, label_tensor.numel() // 10))
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=False,
            gridspec_kw={"height_ratios": (2, 1)},
        )
        axes[0].plot(idx_np, label_np[:, 0], linewidth=0.8)
        axes[0].set_ylabel("Label")
        axes[0].set_xlabel("Time")
        axes[0].set_title("Validation label stream (order of appearance)")
        axes[0].grid(alpha=0.3, linestyle="--", linewidth=0.5)

        axes[1].hist(label_np[:, 0], bins=bins, color="tab:orange", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Label distribution")
        axes[1].grid(alpha=0.3, linestyle="--", linewidth=0.5)
    else:
        bins = min(50, max(10, label_tensor.shape[0] // 10))
        fig, axes = plt.subplots(
            num_dims,
            2,
            figsize=(12, 3 * num_dims),
            sharex=False,
        )
        for dim in range(num_dims):
            axes[dim, 0].plot(idx_np, label_np[:, dim], linewidth=0.8)
            axes[dim, 0].set_ylabel(f"Label[{dim}]")
            axes[dim, 0].set_xlabel("Time")
            axes[dim, 0].set_title(f"Validation label stream dim {dim}")
            axes[dim, 0].grid(alpha=0.3, linestyle="--", linewidth=0.5)

            axes[dim, 1].hist(label_np[:, dim], bins=bins, color="tab:orange", edgecolor="black", linewidth=0.5)
            axes[dim, 1].set_xlabel(f"Label[{dim}]")
            axes[dim, 1].set_ylabel("Count")
            axes[dim, 1].set_title(f"Label[{dim}] distribution")
            axes[dim, 1].grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved label stream visualization to {output_path}")