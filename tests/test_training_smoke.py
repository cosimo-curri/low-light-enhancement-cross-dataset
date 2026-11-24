from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.low_light_enhancement.framework.checkpointing import load_checkpoint
from src.low_light_enhancement.framework.config import (
    build_resolved_configs,
    load_experiment_config,
)
from src.low_light_enhancement.framework.logging import JsonlLogger, TextLogger
from src.low_light_enhancement.framework.trainer import ExperimentTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_one_epoch_training_checkpoint_and_validation(tmp_path, monkeypatch) -> None:
    """Small local integration smoke test; not scientific reproduction."""
    torch.set_num_threads(2)
    monkeypatch.chdir(tmp_path)
    samples = tmp_path / "images"
    samples.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "validation")):
        low = np.random.default_rng(index).integers(0, 220, (32, 32, 3), dtype=np.uint8)
        high = np.minimum(low.astype(np.uint16) + 20, 255).astype(np.uint8)
        input_path = samples / f"low_{index}.png"
        target_path = samples / f"high_{index}.png"
        Image.fromarray(low, mode="RGB").save(input_path)
        Image.fromarray(high, mode="RGB").save(target_path)
        rows.append({
            "dataset": "synthetic", "split": split,
            "input_path": input_path.relative_to(tmp_path).as_posix(),
            "target_path": target_path.relative_to(tmp_path).as_posix(),
            "category": "", "illumination_level": "",
        })

    with Path("train.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = load_experiment_config(ROOT / "configs/selected/unet.yml")
    config = build_resolved_configs(selected)[0]
    config["experiment"]["name"] = "smoke_test"
    config["model"]["channels"] = [8, 16]
    config["data"]["train_manifest"] = "train.csv"
    config["data"]["test_manifests"] = []
    config["augmentation"] = {"enabled": False}
    config["training"].update({
        "epochs": 1, "batch_size": 1, "num_workers": 0,
        "mixed_precision": False,
    })

    log_path = Path("logs/smoke_test/1.jsonl")
    logger = JsonlLogger(log_path)
    try:
        trainer = ExperimentTrainer(
            config, run_index=1, logger=logger,
            text_logger=TextLogger(), progress_bar=False,
        )
        trainer.run()
    finally:
        logger.close()

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_end"
    assert events[-1]["training_finished"] is True
    assert any(event["event"] == "validation_metrics" for event in events)
    checkpoint = load_checkpoint(
        Path("checkpoints/smoke_test/1.pt"),
        map_location=torch.device("cpu"),
    )
    assert checkpoint["training_state"]["training_finished"] is True
    assert checkpoint["random_state"]
    assert checkpoint["best_model_state_dict"]
