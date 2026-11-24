# Experiment provenance and interpretation

## Historical protocol

The original university experiments trained on the LOL-v2 Real train split and selected configurations using only validation results. Each of the four model families was screened with 144 seed-0 configurations (576 training runs). The top three configurations per family were trained with seeds 1, 2, and 3 (36 training runs). The selected configuration per family was evaluated using three trained seed checkpoints (12 final evaluation runs). Thus **624** archived JSONL run files existed in the source project.

Selected configuration IDs from the archived final evaluation `run_start` records, checked against the historical selection hash function:

| Family | Historical selection configuration ID | Local single-run YAML |
|---|---|---|
| U-Net | `d611afe91771` | `configs/selected/unet.yml` |
| Illumination-guided U-Net | `dd999be1c481` | `configs/selected/illumination_guided_unet.yml` |
| FAC-ECL | `c368d8c386cb` | `configs/selected/fac_ecl.yml` |
| ECL-FAR | `c846d880b905` | `configs/selected/ecl_far.yml` |

The selected YAML files preserve the resolved model, optimizer, loss, augmentation and training settings. They are set to **retrain seed 1** and have `evaluation.run_test: false` to avoid treating a newly trained run as the original test set result. The optimizer and model hyperparameters have not been re-selected using test datasets. The original `configs/experiments/` files preserve the full screening grids.

## Historical evaluation

- LOL-v2 Real: in-domain paired test.
- LOL-v2 Synthetic: paired cross-dataset test.
- MILLs: paired cross-dataset test with illumination-level grouping.
- ExDark: **unpaired** real-world stress test; full-reference SSIM/PSNR are not applicable.
- LOL-v1: non-independent overlap-control evaluation; exclude from generalization claims.

The `parsed_logs/*_final_eval.csv` files contain seed-aggregated historical results. These figures describe the archived runs and are **not results of new evaluations**. See the main README for a compact comparison, including important cases where the rankings change by dataset.

### Results archive status

The repository includes the **624-run JSONL corpus** in `experiment_logs/`, the aggregated experiment CSVs and all four per-image failure-candidate JSONs in `parsed_logs/`. Larger qualitative image exports can be generated locally under `qualitative/`; these are not included in the distributed repository. Raw and preprocessed dataset photographs are acquired separately from their publishers, as described in the [dataset setup guide](../README.md#obtain-and-place-the-datasets).

The archived run logs and per-image metadata are included in this repository. Locally generated results and downloaded data must be kept out of commits unless deliberately cleared for publication. Redistribution of dataset-derived metadata and photo-based report panels is subject to the relevant third-party rights: see [`publication-rights.md`](publication-rights.md).

For new executions, logs are created under `logs/`, checkpoint files under `checkpoints/`, and exports under `qualitative/`. These new results must not be confused with previously reported evaluations.


### Archived U-Net CSV schema note

The historical U-Net aggregate CSVs omit the empty `family_specific_config` column that the current common parser includes for all families. Re-parsing the 624-run historical corpus with the original source produces this extra **empty** U-Net column. All other column values and all data rows match the preserved U-Net CSVs exactly. The original archived CSVs were retained unchanged instead of silently normalizing their schema.

## Limitations

The original runs used mixed precision on Kaggle with an NVIDIA Tesla P100, with `training.deterministic: false`. Hardware, CUDA/kernel availability, environment and stochastic augmentation can affect training and metric agreement; matching a seed is not proof of numerical identity. New runs can differ numerically from the archived experiment results. Validate data preprocessing, configuration correctness, model behavior and scientific reproducibility separately.
