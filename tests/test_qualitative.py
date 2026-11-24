from __future__ import annotations

import csv

import pytest

from scripts.export_variant_comparison_panel import load_selected_rows, reset_output_dir


def test_output_dir_is_never_overwritten_implicitly(tmp_path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "result.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        reset_output_dir(destination)
    assert (destination / "result.txt").read_text() == "preserve"
    reset_output_dir(destination, overwrite=True)
    assert list(destination.iterdir()) == []


def test_output_symlink_is_refused(tmp_path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    link = tmp_path / "link"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reset_output_dir(link, overwrite=True)
    assert other.is_dir()


def test_seed_repeats_dataset_sampling(tmp_path) -> None:
    path = tmp_path / "test.csv"
    with path.open("w", encoding="utf8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "input_path", "target_path"])
        writer.writeheader()
        for index in range(20):
            writer.writerow({"split": "test", "input_path": f"image_{index}.png", "target_path": ""})
    first = load_selected_rows(tmp_path, seed=42)
    second = load_selected_rows(tmp_path, seed=42)
    assert first == second
    assert len(first) == 1
