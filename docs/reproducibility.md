# Reproducibility and execution guide

All commands below are issued from the **repository root**. This convention is required by the original `src.low_light_enhancement` import layout and relative paths recorded in the CSV manifests. The historical runtime used Kaggle, a Tesla P100, PyTorch CUDA and Python with the dependencies pinned in `requirements.txt`. The exact original Python version and complete transitive dependency lock were not recorded; no guarantee of cross-platform binary equivalence is made.

## Installation

Use **Python 3.13**. The pinned `requirements.txt` installs PyTorch 2.12.0 from a **CUDA 12.6** wheel source. Check the operating system, NVIDIA driver and CUDA compatibility before installing:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

If your hardware does not support this wheel, adapt the PyTorch installation for your platform explicitly and record the change. For unit tests, also install `requirements-dev.txt` (`pytest`). Some evaluation metrics use `pyiqa`, which may download additional pretrained metric models; plan for connectivity and licensing of those assets.

## Data setup

The complete list of official dataset download URLs and exact raw/processed directory trees is in [`README.md` → Datasets](../README.md#obtain-and-place-the-datasets). Download the archives from their publishers and organize the source images under `data/raw/` as required by `configs/data/datasets.yml`. Dataset photographs are not included in Git and no unattended downloader is implemented.

```bash
python scripts/build_manifests.py --config configs/data/datasets.yml
python scripts/preprocess_offline.py --config configs/data/preprocess.yml --no-progress-bar
```

The first command produces five manifest CSVs under `data/manifests/`. The second creates their processed CSV versions, a `dataset_summary.csv`, and 256 × 256 PNGs in `data/preprocessed/`. The repository includes the five original raw-data manifest CSVs, five processed CSVs and `dataset_summary.csv`, but not the raw or processed images. Do not stage downloaded images or generated outputs when committing project changes. After downloading raw images, regenerate the preprocessed image outputs, and rebuild the CSVs if the downloaded image inventory or paths differ.

Preprocessing applies EXIF orientation, RGB conversion and LANCZOS resize; image statistics are measured before resizing. The LOL-v2 Real train/validation split is seeded with 42. When reusing already processed data, ensure the CSV `input_path` / `target_path` fields resolve to the actual image files; a CSV alone is not a runnable dataset. A new clone can regenerate the CSVs from the original image archives, but matching published benchmark values is not guaranteed without the original conditions.

## Inspecting and running models

The full screening grids are preserved; **do not launch one by accident**. Inspect expansion counts without training:

```bash
python scripts/start_experiments.py --config configs/experiments/unet.yml --dry-run
# 144 resolved runs, no training
python scripts/start_experiments.py --config configs/selected/unet.yml --dry-run
# One selected training run, no training
```

To train one of the historically selected configurations with a fresh seed-1 run:

```bash
python scripts/start_experiments.py --config configs/selected/unet.yml --no-progress-bar
```

To train a different family, replace `unet` with `illumination_guided_unet`, `fac_ecl` or `ecl_far`. These settings include up to 999 epochs, early stopping and GPU-heavy architecture choices; **single-run does not mean quick**. `evaluation.run_test: false` is deliberate. Do not automatically change it to `true` during model-selection or claim results comparable to the historical paper without checking the complete evaluation protocol.

## Parsing JSONL logs

The archived JSONL logs are included in `experiment_logs/` and tracked in Git. To parse those or another local log directory:

```bash
python scripts/parse_logs.py --parsing-mode screening \\
  --logs-dir logs/<experiment-name> \\
  --output-name new_screening.csv
```

The supported modes are `screening`, `multi_seed`, `final_eval`, `failure_candidates`; they require the respective event structures. Outputs are written to `parsed_logs/`. Use a new output name, rather than overwriting archived aggregate CSVs.

## Export and checkpoint security

New checkpoint files are saved in `checkpoints/`. Fresh checkpoints now store the NumPy RNG state as primitive Python values while preserving its state contents, permitting PyTorch `weights_only=True` loading. Legacy checkpoint files containing pickled NumPy arrays are not loaded by default. **Only if you have independently verified ownership and trust of a legacy file**, use the relevant `--trust-legacy-checkpoint` export flag or set `experiment.trust_legacy_checkpoint: true` for trainer restore. This deliberately opts into unsafe pickle loading. Never enable it for downloaded, untrusted weights.

```bash
python scripts/export_qualitative_examples.py \\
  --checkpoint checkpoints/<experiment-name>/1.pt \\
  --candidates <path/to/legitimate_candidates.json>
```

The four-family comparison requires all four checkpoint paths, local test manifests and images. It now supports `--seed` for repeatable sample selection and will refuse to remove an existing nonempty output directory unless `--overwrite` is passed.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The automated suite exercises configuration expansion and selection, preprocessing on generated images, checkpoint RNG round-tripping, security defaults, model interfaces and export safeguards. Use the experiment configuration files and dataset preparation instructions above to run training and evaluation.

## Kaggle

`run_experiments_kaggle.ipynb` contains the equivalent opt-in stages. It lists publisher download links and the exact folder layout. Provide the URL of the published repository and mount separately acquired datasets as Kaggle Inputs. Its default settings run **no** experiments and do not download any raw image datasets automatically.
