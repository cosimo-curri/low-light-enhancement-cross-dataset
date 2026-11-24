from __future__ import annotations

from pathlib import Path

import pytest

from src.low_light_enhancement.framework.config import (
    build_resolved_configs,
    build_selection_config_hash,
    load_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["unet", "illumination_guided_unet", "fac_ecl", "ecl_far"]
EXPECTED_SELECTION_IDS = {
    "unet": "d611afe91771",
    "illumination_guided_unet": "dd999be1c481",
    "fac_ecl": "c368d8c386cb",
    "ecl_far": "c846d880b905",
}


@pytest.mark.parametrize("family", FAMILIES)
def test_historical_screening_grid_count(family: str) -> None:
    config = load_experiment_config(ROOT / "configs/experiments" / f"{family}.yml")
    expanded = build_resolved_configs(config)
    assert len(expanded) == 144
    assert {run["experiment"]["seed"] for run in expanded} == {0}
    assert all(run["evaluation"]["run_test"] is False for run in expanded)


@pytest.mark.parametrize("family", FAMILIES)
def test_selected_configuration_is_one_run_and_matches_historical_hash(family: str) -> None:
    config = load_experiment_config(ROOT / "configs/selected" / f"{family}.yml")
    runs = build_resolved_configs(config)
    assert len(runs) == 1
    resolved = runs[0]
    assert resolved["experiment"]["seed"] == 1
    assert "restore_checkpoint" not in resolved["experiment"]
    assert resolved["evaluation"]["run_test"] is False
    assert build_selection_config_hash(resolved) == EXPECTED_SELECTION_IDS[family]


def test_hash_ignores_name_and_evaluation_but_not_model() -> None:
    config = build_resolved_configs(load_experiment_config(
        ROOT / "configs/selected/unet.yml"
    ))[0]
    original_hash = build_selection_config_hash(config)
    config["experiment"]["name"] = "different_run_name"
    config["evaluation"]["run_test"] = True
    assert build_selection_config_hash(config) == original_hash
    config["optimizer"]["lr"] *= 2
    assert build_selection_config_hash(config) != original_hash
