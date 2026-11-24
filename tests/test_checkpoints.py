from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.low_light_enhancement.framework.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from src.low_light_enhancement.framework.torch_utils import (
    get_random_state,
    set_random_state,
)


def test_safe_checkpoint_roundtrip_and_rng_continuity(tmp_path) -> None:
    random.seed(129)
    np.random.seed(129)
    torch.manual_seed(129)
    state = get_random_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    save_checkpoint(tmp_path / "safe.pt", {"random_state": state, "tensor": torch.tensor([1.0])})
    random.random()
    np.random.rand()
    torch.rand(3)

    checkpoint = load_checkpoint(tmp_path / "safe.pt", map_location=torch.device("cpu"))
    set_random_state(checkpoint["random_state"])
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_unsafe_legacy_pickle_requires_explicit_opt_in(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"legacy_array": np.array([1, 2], dtype=np.uint32)}, path)
    with pytest.raises(ValueError, match="weights_only=True"):
        load_checkpoint(path, map_location=torch.device("cpu"))
    loaded = load_checkpoint(path, map_location=torch.device("cpu"), trusted_legacy=True)
    assert np.array_equal(loaded["legacy_array"], np.array([1, 2], dtype=np.uint32))
