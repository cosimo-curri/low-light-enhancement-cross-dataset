from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device
) -> dict[str, Any]:
    moved_batch = dict(batch)
    moved_batch["input"] = batch["input"].to(device, non_blocking=True)

    if batch["target"] is not None:
        moved_batch["target"] = batch["target"].to(device, non_blocking=True)

    return moved_batch


def set_random_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    # Store the NumPy uint32 buffer as plain integers. This preserves the RNG
    # state while allowing the checkpoint to be loaded with weights_only=True.
    serializable_numpy_state = (
        numpy_state[0],
        numpy_state[1].tolist(),
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4])
    )
    random_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": serializable_numpy_state,
        "torch": torch.get_rng_state()
    }

    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()

    return random_state


def set_random_state(random_state: dict[str, Any]) -> None:
    random.setstate(random_state["python"])
    numpy_state = random_state["numpy"]
    # Accept both the original NumPy array representation and the new safe
    # list representation, keeping RNG restoration semantically identical.
    numpy_state = (
        numpy_state[0],
        np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4]
    )
    np.random.set_state(numpy_state)

    torch_state = random_state["torch"]

    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.cpu()

    torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and "cuda" in random_state:
        cuda_states = [
            state.cpu() if isinstance(state, torch.Tensor) else state
            for state in random_state["cuda"]
        ]

        torch.cuda.set_rng_state_all(cuda_states)