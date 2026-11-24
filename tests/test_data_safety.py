"""Regression tests for safe data preparation and explicit input validation."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.build_manifests import build_dataset_rows, list_image_files, main as manifest_main
from scripts.preprocess_offline import main as preprocess_main
from src.low_light_enhancement.framework.checkpointing import load_checkpoint


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 8), color=(20, 60, 110)).save(path)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_only_supported_images_and_absent_folders_raise(tmp_path: Path) -> None:
    _image(tmp_path / "low" / "10.PNG")
    _write(tmp_path / "low" / "README.txt", "not an image")
    assert [p.name for p in list_image_files(tmp_path / "low")] == ["10.PNG"]
    with pytest.raises(FileNotFoundError, match="Required dataset directory"):
        list_image_files(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No supported images"):
        list_image_files(tmp_path / "empty")


def test_missing_pair_names_source_and_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _image(tmp_path / "data/raw/pairs/train/low/one.png")
    _image(tmp_path / "data/raw/pairs/train/high/two.png")
    with pytest.raises(ValueError, match="Missing paired target 'one.png'"):
        build_dataset_rows(
            "pairs",
            {
                "root": "pairs",
                "splits": {"train": {"input_dir": "train/low", "target_dir": "train/high"}},
                "pairing": {"strategy": "same_filename"},
            },
            tmp_path / "data/raw",
        )


def test_missing_later_dataset_does_not_overwrite_first_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _image(tmp_path / "data/raw/first/class_a/one.png")
    csv_path = tmp_path / "data/manifests/first.csv"
    _write(csv_path, "ORIGINAL_MANIFEST\n")
    _write(
        tmp_path / "config.yml",
        """manifests:
  raw_dir: data/raw
  output_dir: data/manifests
datasets:
  first:
    root: first
    output: first.csv
    splits:
      test:
        input_dir: .
    pairing:
      strategy: category_dirs
  second:
    root: second
    output: second.csv
    splits:
      test:
        input_dir: .
    pairing:
      strategy: category_dirs
""",
    )
    monkeypatch.setattr(sys, "argv", ["build_manifests.py", "--config", "config.yml"])
    with pytest.raises(FileNotFoundError, match="second"):
        manifest_main()
    assert csv_path.read_text(encoding="utf-8") == "ORIGINAL_MANIFEST\n"
    assert not (tmp_path / "data/manifests/second.csv").exists()


def test_refuse_reduced_inventory_without_explicit_option(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _image(tmp_path / "data/raw/first/class_a/one.png")
    csv_path = tmp_path / "data/manifests/first.csv"
    _write(csv_path, "dataset,split,input_path,target_path,category,illumination_level\nfirst,test,a,,class_a,\nfirst,test,b,,class_a,\n")
    _write(
        tmp_path / "config.yml",
        """manifests:
  raw_dir: data/raw
  output_dir: data/manifests
datasets:
  first:
    root: first
    output: first.csv
    splits:
      test:
        input_dir: .
    pairing:
      strategy: category_dirs
""",
    )
    original_bytes = csv_path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["build_manifests.py", "--config", "config.yml"])
    with pytest.raises(ValueError, match="Refusing to replace"):
        manifest_main()
    assert csv_path.read_bytes() == original_bytes
    monkeypatch.setattr(sys, "argv", ["build_manifests.py", "--config", "config.yml", "--allow-smaller-manifests"])
    manifest_main()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_preprocessing_missing_images_never_overwrites_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write(
        tmp_path / "config.yml",
        """preprocessing:
  input_manifest_dir: data/manifests
  output_dir: data/preprocessed
  image:
    size: [16, 16]
    output_format: png
  statistics:
    dark_threshold: 0.1
""",
    )
    _write(
        tmp_path / "data/manifests/first.csv",
        "dataset,split,input_path,target_path,category,illumination_level\nfirst,test,data/raw/first/absent.png,,class_a,\n",
    )
    output_path = tmp_path / "data/preprocessed/first.csv"
    _write(output_path, "ORIGINAL_PROCESSED_MANIFEST\n")
    monkeypatch.setattr(sys, "argv", ["preprocess_offline.py", "--config", "config.yml", "--no-progress-bar"])
    with pytest.raises(FileNotFoundError, match="data/raw/first/absent.png"):
        preprocess_main()
    assert output_path.read_text(encoding="utf-8") == "ORIGINAL_PROCESSED_MANIFEST\n"


def test_missing_checkpoint_reports_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", map_location=torch.device("cpu"))
