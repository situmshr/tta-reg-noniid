import os
import random
from contextlib import contextmanager

import torch
import torch.backends.cudnn as torch_backends_cudnn

import numpy as np


def fix_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_backends_cudnn.benchmark = False


@contextmanager
def preserve_rng_state():
    """Temporarily preserve Python/NumPy/Torch RNG states."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
