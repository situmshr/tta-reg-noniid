# import numpy as np
# import torch
# from torch.utils.data import TensorDataset

# def get_note_non_iid_dataset(dataset, num_chunks=10, beta=0.1, min_chunk_size=10, num_bins=10, seed=None):
#     rng = np.random.default_rng(seed)

#     X, y = [], []
#     for feat, label in dataset:
#         X.append(feat if isinstance(feat, torch.Tensor) else torch.tensor(feat))
#         y.append(float(label))
#     X = torch.stack(X)
#     y = torch.tensor(y)

#     if y.dtype.is_floating_point and len(torch.unique(y)) > num_bins:
#         bins = np.linspace(y.min(), y.max(), num_bins + 1)
#         y_indices = np.digitize(y.numpy(), bins[1:-1])
#     else:
#         y_indices = y.numpy().astype(int)

#     N = len(y)
#     limit_per_chunk = N / num_chunks

#     while True:
#         chunks = [[] for _ in range(num_chunks)]
#         chunk_sizes = np.zeros(num_chunks)

#         for c in np.unique(y_indices):
#             indices = np.where(y_indices == c)[0]
#             rng.shuffle(indices)

#             props = rng.dirichlet(np.repeat(beta, num_chunks))

#             mask = (chunk_sizes < limit_per_chunk)
#             if not mask.any(): mask[:] = True
#             props *= mask
#             props /= props.sum()

#             split_pts = (np.cumsum(props) * len(indices)).astype(int)[:-1]
#             for i, split_idx in enumerate(np.split(indices, split_pts)):
#                 if len(split_idx) > 0:
#                     chunks[i].append(split_idx)
#                     chunk_sizes[i] += len(split_idx)

#         if chunk_sizes.min() >= min_chunk_size:
#             break

#     final_indices = []
#     for chunk in chunks:
#         rng.shuffle(chunk)
#         final_indices.extend([idx for sublist in chunk for idx in sublist])
    
#     return TensorDataset(X[final_indices], y[final_indices])

import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_note_non_iid_dataset(dataset, num_chunks=10, beta=0.1, transition_ratio=0.3, min_chunk_size=10, num_bins=10, seed=None):
    rng = np.random.default_rng(seed)

    X, y = [], []
    for feat, label in dataset:
        X.append(feat if isinstance(feat, torch.Tensor) else torch.tensor(feat))
        y.append(float(label))
    X = torch.stack(X)
    y = torch.tensor(y)

    if y.dtype.is_floating_point and len(torch.unique(y)) > num_bins:
        bins = np.linspace(y.min(), y.max(), num_bins + 1)
        y_indices = np.digitize(y.numpy(), bins[1:-1])
    else:
        y_indices = y.numpy().astype(int)
    # if y.dtype.is_floating_point and len(torch.unique(y)) > num_bins:
    #     cutoff = 60.0
    #     if y.max() >= cutoff:
    #         bins = np.linspace(y.min(), cutoff, num_bins)
    #         y_indices = np.digitize(y.numpy(), bins[1:])  # bucket ages 60+ together
    #     else:
    #         bins = np.linspace(y.min(), y.max(), num_bins + 1)
    #         y_indices = np.digitize(y.numpy(), bins[1:-1])
    # else:
    #     y_indices = y.numpy().astype(int)

    N = len(y)
    limit_per_chunk = N / num_chunks

    while True:
        chunks = [[] for _ in range(num_chunks)]
        chunk_sizes = np.zeros(num_chunks)

        for c in np.unique(y_indices):
            indices = np.where(y_indices == c)[0]
            rng.shuffle(indices)

            props = rng.dirichlet(np.repeat(beta, num_chunks))

            mask = (chunk_sizes < limit_per_chunk)
            if not mask.any(): mask[:] = True
            props *= mask
            props /= props.sum()

            split_pts = (np.cumsum(props) * len(indices)).astype(int)[:-1]
            for i, split_idx in enumerate(np.split(indices, split_pts)):
                if len(split_idx) > 0:
                    chunks[i].append(split_idx)
                    chunk_sizes[i] += len(split_idx)

        if chunk_sizes.min() >= min_chunk_size:
            break

    ordered_blocks = []
    for chunk in chunks:
        for sublist in chunk:
            block = [idx for idx in sublist]
            rng.shuffle(block)
            ordered_blocks.append(block)

    final_indices = []
    buffer = []

    for i, block in enumerate(ordered_blocks):
        n = len(block)
        
        n_mix_head = int(n * transition_ratio) if i > 0 else 0
        n_mix_tail = int(n * transition_ratio) if i < len(ordered_blocks) - 1 else 0
        
        if n_mix_head + n_mix_tail > n:
            scale = n / (n_mix_head + n_mix_tail)
            n_mix_head = int(n_mix_head * scale)
            n_mix_tail = int(n_mix_tail * scale)

        head = block[:n_mix_head]
        body = block[n_mix_head : n - n_mix_tail]
        tail = block[n - n_mix_tail:]

        if i > 0:
            mix_indices = []
            len_prev = len(buffer)
            len_curr = len(head)
            total_mix_len = len_prev + len_curr
            
            probs = np.linspace(0, 1, total_mix_len)
            
            ptr_prev = 0
            ptr_curr = 0
            
            for p in probs:
                use_curr = rng.random() < p
                
                if (use_curr and ptr_curr < len_curr) or ptr_prev >= len_prev:
                    mix_indices.append(head[ptr_curr])
                    ptr_curr += 1
                else:
                    mix_indices.append(buffer[ptr_prev])
                    ptr_prev += 1
            
            final_indices.extend(mix_indices)

        final_indices.extend(body)
        buffer = tail

    return TensorDataset(X[final_indices], y[final_indices])
