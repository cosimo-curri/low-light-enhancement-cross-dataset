from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_checkpoint_path(experiment_name: str, run_index: int) -> Path:
    return Path("checkpoints") / experiment_name / f"{run_index}.pt"


def save_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: torch.device,
    trusted_legacy: bool = False
) -> dict[str, Any]:
    """Load safe tensor/primitive checkpoints by default.

    Historical checkpoints can contain NumPy objects requiring pickle.
    Use trusted_legacy=True *only* for files produced by you and whose
    provenance you independently trust. Arbitrary pickle can execute code.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True
        )
    except OSError:
        # Missing paths and permission errors are not deserialization failures.
        raise
    except Exception as error:
        if not trusted_legacy:
            raise ValueError(
                "Checkpoint could not be loaded with weights_only=True. "
                "Do not load untrusted checkpoint files. For a verified "
                "legacy checkpoint only, call load_checkpoint(..., "
                "trusted_legacy=True)."
            ) from error
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    return checkpoint
