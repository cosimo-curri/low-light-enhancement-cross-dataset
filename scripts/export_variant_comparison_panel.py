from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bootstrap import add_project_root_to_path


add_project_root_to_path()  # allows imports from src


from src.low_light_enhancement.framework.checkpointing import load_checkpoint
from src.low_light_enhancement.framework.io import (
    read_csv_rows,
    read_image_tensor
)
from src.low_light_enhancement.framework.qualitative import (
    build_output_prefix,
    make_panel
)
from src.low_light_enhancement.framework.registry import build_model_wrapper
from src.low_light_enhancement.framework.torch_utils import get_device
from src.low_light_enhancement.framework.transforms import tensor_to_image


MANIFESTS_DIR = Path("data") / "manifests"
OUTPUT_DIR = Path("qualitative") / "inter_family_random_panels"


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class LoadedEnhancer:
    label: str
    checkpoint_path: Path
    config: dict[str, Any]
    wrapper: Any
    model: torch.nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export random variant-comparison qualitative panels."
    )

    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        required=True,
        help="Path to the selected U-Net checkpoint."
    )

    parser.add_argument(
        "--illumination-guided-unet-checkpoint",
        type=Path,
        required=True,
        help="Path to the selected illumination-guided U-Net checkpoint."
    )

    parser.add_argument(
        "--fac-ecl-checkpoint",
        type=Path,
        required=True,
        help="Path to the selected FAC-ECL checkpoint."
    )

    parser.add_argument(
        "--ecl-far-checkpoint",
        type=Path,
        required=True,
        help="Path to the selected ECL-FAR checkpoint."
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for reproducible sample selection (default: system randomness)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly allow removing previous panels in the output directory."
    )
    parser.add_argument(
        "--trust-legacy-checkpoint", action="store_true",
        help="Allow unsafe pickle only for independently trusted legacy checkpoints."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    enhancers = [
        load_enhancer("U-Net", args.unet_checkpoint, device, args.trust_legacy_checkpoint),
        load_enhancer(
            "illumination-guided U-Net",
            args.illumination_guided_unet_checkpoint,
            device,
            args.trust_legacy_checkpoint
        ),
        load_enhancer("FAC-ECL", args.fac_ecl_checkpoint, device, args.trust_legacy_checkpoint),
        load_enhancer("ECL-FAR", args.ecl_far_checkpoint, device, args.trust_legacy_checkpoint)
    ]

    selected_rows = load_selected_rows(MANIFESTS_DIR, seed=args.seed)

    reset_output_dir(OUTPUT_DIR, overwrite=args.overwrite)

    for index, row in enumerate(selected_rows, start=1):
        export_variant_comparison_panel(
            index=index,
            row=row,
            enhancers=enhancers,
            output_dir=OUTPUT_DIR,
            device=device
        )

    print(
        f"Exported {len(selected_rows)} variant-comparison panel(s) to: "
        f"{OUTPUT_DIR}"
    )


def load_enhancer(
    label: str,
    checkpoint_path: Path,
    device: torch.device,
    trusted_legacy: bool = False
) -> LoadedEnhancer:
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
        trusted_legacy=trusted_legacy
    )

    config = checkpoint["resolved_config"]

    wrapper = build_model_wrapper(config["model"]["name"])

    model = wrapper.build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["best_model_state_dict"])
    model.eval()

    return LoadedEnhancer(
        label=label,
        checkpoint_path=checkpoint_path,
        config=config,
        wrapper=wrapper,
        model=model
    )


def load_selected_rows(
    manifests_dir: Path,
    *,
    seed: int | None = None
) -> list[dict[str, str]]:
    selected_rows: list[dict[str, str]] = []
    rng = random.Random(seed) if seed is not None else random.SystemRandom()

    for manifest_path in sorted(manifests_dir.glob("*.csv")):
        rows = load_manifest_test_rows(manifest_path)
        selected_rows.append(rng.choice(rows))

    return selected_rows


def load_manifest_test_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_inputs: set[str] = set()
    manifest_rows, _ = read_csv_rows(manifest_path)

    for row in manifest_rows:
        if row.get("split") != "test":
            continue

        input_path = row["input_path"]

        if input_path in seen_inputs:
            continue

        seen_inputs.add(input_path)
        rows.append(row)

    return rows


def reset_output_dir(output_dir: Path, *, overwrite: bool = False) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite to replace previous exported panels."
            )
        for path in output_dir.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def export_variant_comparison_panel(
    *,  # next options are keyword-only
    index: int,
    row: dict[str, str],
    enhancers: list[LoadedEnhancer],
    output_dir: Path,
    device: torch.device
) -> None:
    input_tensor = read_image_tensor(Path(row["input_path"]))
    input_batch = input_tensor.unsqueeze(0).to(device)

    panel_images = [tensor_to_image(input_tensor)]
    panel_labels = ["Input"]

    with torch.no_grad():
        batch = {"input": input_batch}

        for enhancer in enhancers:
            output = enhancer.wrapper.forward(enhancer.model, batch)
            prediction = enhancer.wrapper.get_prediction(output)[0]

            panel_images.append(tensor_to_image(prediction))
            panel_labels.append(enhancer.label)

    target_path = row.get("target_path", "")

    if target_path:
        target_tensor = read_image_tensor(Path(target_path))

        panel_images.append(tensor_to_image(target_tensor))
        panel_labels.append("Target")

    panel = make_panel(
        images=panel_images,
        labels=panel_labels
    )

    output_path = output_dir / f"{build_output_prefix(index, row)}_panel.png"
    panel.save(output_path)


if __name__ == "__main__":
    main()