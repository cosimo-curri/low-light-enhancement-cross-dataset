from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.low_light_enhancement.framework.config import (
    build_resolved_configs,
    load_experiment_config,
)
from src.low_light_enhancement.framework.registry import build_model_wrapper
from src.low_light_enhancement.framework.torch_utils import count_parameters

ROOT = Path(__file__).resolve().parents[1]
FAMILY_PARAMS = [
    ("unet", 7763107),
    ("illumination_guided_unet", 1927363),
    ("fac_ecl", 1927075),
    ("ecl_far", 2042471),
]


@pytest.mark.parametrize("family,expected_parameters", FAMILY_PARAMS)
def test_model_output_and_parameter_count(family: str, expected_parameters: int) -> None:
    torch.set_num_threads(2)
    config = build_resolved_configs(load_experiment_config(
        ROOT / "configs/selected" / f"{family}.yml"
    ))[0]
    wrapper = build_model_wrapper(config["model"]["name"])
    model = wrapper.build_model(config["model"])
    model.eval()
    assert count_parameters(model) == expected_parameters
    with torch.no_grad():
        output = wrapper.forward(model, {"input": torch.rand(1, 3, 32, 32)})
        prediction = wrapper.get_prediction(output)
    assert prediction.shape == (1, 3, 32, 32)
    assert torch.isfinite(prediction).all()
