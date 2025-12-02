import numpy as np
import torch
from torch.utils.data import TensorDataset


def get_non_iid_dataset(
    dataset,
    mode: str = "linear",   # "linear" or "sine"
    sigma: float = 1.0,
    cycles: int = 1,
    length: int | None = None,
    seed: int | None = None,
):
    """
    Create a Non-IID data stream using rank-based reordering.

    Common steps:
        - Extract all (X, y) from the dataset.
        - Sort samples by label y to get indices sorted_idx (ranks 0..N-1).
        - Build a permutation of these ranks according to `mode`.
        - Apply that permutation to obtain the final stream order.

    mode="linear":
        - Each sample k (in sorted-by-y order) gets a base time:
              base_t[k] = k / N
        - Add Gaussian time noise eps ~ N(0, sigma^2):
              t_k = base_t[k] + eps_k
        - Sort by t_k; this order defines the stream.
        - sigma controls local mixing around the monotone trend.
          (sigma=0 → perfectly sorted by label)

    mode="sine":
        - For each time step t = 0..N-1, define a target quantile u_t in [0,1]
          following a sine wave with `cycles` cycles over the stream:
              u_t = 0.5 * (1 + sin(2π * cycles * t/N - π/2))
        - Optionally add noise N(0, sigma^2) to u_t (clipped to [0,1]) for
          local mixing.
        - Sort u_t; smallest u_t gets rank 0, next gets rank 1, ..., largest
          gets rank N-1. This gives a one-to-one mapping time → rank.
        - Map ranks to sorted_idx to obtain the stream order.
        - The expected label over time follows a sine-like up/down pattern.

    Args:
        dataset: PyTorch Dataset with (feat, label) pairs.
        mode: "linear" or "sine".
        sigma: noise scale.
            - linear: std dev of time noise t_k.
            - sine  : std dev of quantile noise u_t.
        cycles: number of sine cycles across the full stream (sine mode only).
        length: optional length of returned stream (<= N). If None, use all N.
        seed: random seed.

    Returns:
        TensorDataset with samples reordered into a Non-IID stream.
    """
    rng = np.random.default_rng(seed)

    # 1. Extract all samples
    X_list, y_list = [], []
    for feat, label in dataset:
        X_list.append(feat if isinstance(feat, torch.Tensor) else torch.tensor(feat))
        y_list.append(float(label))
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)

    N = len(y)
    if length is None:
        length = N
    if length > N:
        raise ValueError(f"length ({length}) must be <= number of samples ({N})")

    y_np = y.numpy()
    # 2. Sort by label (ascending) → define ranks 0..N-1
    sorted_idx = np.argsort(y_np)

    if mode == "linear":
        # base time per sample in rank space
        base_t = np.linspace(0.0, 1.0, N, endpoint=False)
        eps = rng.normal(loc=0.0, scale=sigma, size=N)
        t = base_t + eps
        t = np.clip(t, 0.0, 1.0)
        order = np.argsort(t)            # permutation of ranks 0..N-1
        stream_idx = sorted_idx[order]

    elif mode == "sine":
        # normalized time grid
        t_grid = np.linspace(0.0, 1.0, N, endpoint=False)
        # target quantiles following a sine wave in [0, 1]
        u = 0.5 * (1.0 + np.sin(2.0 * np.pi * cycles * t_grid - np.pi / 2.0))
        if sigma > 0.0:
            u = u + rng.normal(loc=0.0, scale=sigma, size=N)
            u = np.clip(u, 0.0, 1.0)

        # assign ranks to times by sorting u
        order_u = np.argsort(u)          # times in increasing desired quantile
        assigned_rank = np.empty(N, dtype=int)
        for rank, t_idx in enumerate(order_u):
            assigned_rank[t_idx] = rank

        stream_idx = sorted_idx[assigned_rank]

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'linear' or 'sine'.")

    # truncate if shorter stream requested
    stream_idx = stream_idx[:length]
    stream_idx_torch = torch.from_numpy(stream_idx)

    return TensorDataset(X[stream_idx_torch], y[stream_idx_torch])
