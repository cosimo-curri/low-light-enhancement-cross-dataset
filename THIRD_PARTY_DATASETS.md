# Third-party datasets and redistribution

Download raw images from the dataset owners using the official links below; arrange them under `data/raw/` according to the **exact [README directory guide](README.md#obtain-and-place-the-datasets)**. Raw and preprocessed photographs are not included in this repository. The five raw-data CSV manifests in `data/manifests/`, six root-level preprocessing CSVs, archived `experiment_logs/`, and all `parsed_logs/` files are included; `logs/` is reserved for new runs. Keep downloaded images and generated artifacts out of commits.

**Important:** Some CSVs and log events contain per-image identifiers or dataset-derived metadata. Their inclusion in Git is a technical choice, **not** a licence to redistribute third-party rights-protected content. Check relevant source terms, especially ExDark, before publishing. The repository includes five raw-data manifest CSVs and six preprocessing CSVs. The image files are not distributed; the existence of paths within a CSV does not establish permission to redistribute the associated metadata.

| Dataset | Original source | Published access and reuse considerations | Data handling |
|---|---|---|---|
| LOL-v1 | https://daooshee.github.io/BMVC2018website/ | Download offered; no explicit photo redistribution grant found on official project page | Keep dataset-derived photos private pending written permission or a justified exception |
| LOL-v2 | https://github.com/flyywh/CVPR-2020-Semi-Low-Light | Official code/data page does not provide a verified general photo redistribution grant | Keep dataset-derived photos private pending permission/rights review |
| MILL / MILLs | https://huggingface.co/datasets/davidserra9/MILL and https://github.com/CVC-Color/MILL | Hugging Face dataset card states `license: mit`; verify intended files are actually within its scope | Redistribution may be possible if covered; retain licence and credit |
| ExDark | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Dataset Research Use Terms effective 2026-10-01 prohibit unpermitted redistribution of covered annotations, metadata and splits, while third-party photos are expressly not licensed by those terms | Do not publish ExDark photos or protected dataset metadata without relevant permissions |

The ExDark project distinguishes code, annotations/metadata and third-party photographs. Official terms: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Dataset/DATASET_TERMS.md. Its terms explicitly permit aggregate academic results and analyses within permitted non-commercial research, subject to applicable limits. They do **not** authorise republishing third-party photos, annotated indices or arbitrary per-image metadata. The presence of a BSD software license at the repository level does not relicense the data.

A GitHub portfolio is public: attribution alone does not replace copyright permission. The rights to original photos remain with their owners, and transformations or result montages may retain protected content. These conclusions are a conservative review of official source statements, not an individual legal opinion or a guarantee of compliance in every jurisdiction. See [`docs/publication-rights.md`](docs/publication-rights.md) for figure-level implications.

## Local dataset layout

The precise expected directory tree, direct links to downloadable dataset archives, filename-pairing rules, generated CSV names and processing commands are documented in [`README.md` → Datasets](README.md#obtain-and-place-the-datasets). Follow `configs/data/datasets.yml` exactly and invoke scripts from the repository root. Manifest generation checks for missing source images and missing pairs before writing any CSV; it also refuses to reduce existing row counts unless `--allow-smaller-manifests` is provided. Keep a separate backup before intentionally replacing reference CSVs. See `report/main.bib` for research citations.
