# Publication rights and externally sourced assets

This document explains the provenance and potential redistribution constraints of externally sourced materials. It is **not legal advice**; the applicable terms and permissions should be checked with the original publishers and relevant rights holders.

## Repository content and third-party rights

This distribution does not include raw or processed dataset photographs, bulk `qualitative/` exports, model checkpoints or local build artifacts. It does include `data/manifests/*.csv`, `data/preprocessed/*.csv`, the archived `experiment_logs/` and `parsed_logs/` files, `report/main.pdf`, the report figures and the LaTeX style/bibliography inputs. The `logs/` directory is present for future runs.

This inclusion is a file-management decision, not a grant of permission to redistribute protected photographs or associated metadata. The complete report PDF contains third-party image derivatives; logs/CSVs/JSONs may contain per-image paths or dataset-specific annotations. Before public distribution, review actual staged content and assess the relevant rights. The repository includes five raw-data manifest CSVs and six preprocessing CSVs. Their inclusion does **not** establish permission to publish any covered dataset metadata.

Before publishing new commits or archives, inspect `git status`, `git diff --cached --name-only` and all staged or tracked files. Review raw and processed dataset images, local checkpoints, generated exports and temporary files for inclusion risks; respect image copyright and privacy requirements, and stage only the intended materials.

## Original sources and licensing considerations

- **LOL-v1:** https://daooshee.github.io/BMVC2018website/ — offers dataset download but no verified general licence to redistribute its photographic content.
- **LOL-v2:** https://github.com/flyywh/CVPR-2020-Semi-Low-Light — the original repository offers access to images but a broad grant to republish photographs was not verified.
- **MILL/MILLs:** https://huggingface.co/datasets/davidserra9/MILL — dataset card identifies `mit`; check the exact materials under that licence and preserve any required original license/copyright notices. Its licence does not grant rights in different datasets placed in the same montage.
- **ExDark:** https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Dataset/DATASET_TERMS.md — effective 1 October 2026, covered project-created metadata, annotations and splits are for authorised non-commercial research and may not be redistributed without written permission; underlying images often originate from third parties and are not covered by this grant. Aggregate academic results are permitted within scope.
- **CVPR author kit:** https://github.com/cvpr-org/author-kit — `cvpr.sty` and `ieeenat_fullname.bst` originate there. Their exact redistribution licence was not verified from the upstream repository and requires attribution and a separate check before public publication.

## Photo-containing outputs

The source report contains six photo-bearing result figure files:

1. `report/fig/unet_qualitative/unet_qualitative.png`
2. `report/fig/illumination_guided_unet_qualitative/illumination_guided_unet_qualitative.png`
3. `report/fig/fac_ecl_qualitative/fac_ecl_qualitative.png`
4. `report/fig/ecl_far_qualitative/ecl_far_qualitative.png`
5. `report/fig/ecl_far_intermediate/ecl_far_intermediate.png`
6. `report/fig/inter_family_random_panel/inter_family_random_panel.png`

The family galleries contain example inputs, model outputs and targets from several datasets, and the inter-family panel explicitly combines LOL-v1, LOL-v2 Real/Synthetic, MILLs and ExDark. Because at least some source photographs have uncertain/restricted rights, **these full composite images should not be publicly posted without a documented permission or an applicable and carefully assessed exception**. The same concern applies to `qualitative/**/*.png`, and to `report/main.pdf` because it embeds the above figures. The model-produced enhanced image is derived from the source photograph; the enhancement does not automatically extinguish source rights.

Three remaining figures — `report/fig/pipeline/`, `report/fig/experimental_protocol/`, and `report/fig/mills_ssim_by_illumination/` — are methodological diagrams or aggregated chart outputs without obvious third-party photographs, and are suitable for ordinary project documentation subject to ordinary authorship checks.

The original research results, aggregate statistics and source code can be documented without reproducing the underlying photos. Per-image manifests and failure-candidate metadata may also include ExDark-covered metadata; therefore, their public redistribution may require specific permission. Historical JSONL logs can contain per-image paths and evaluation events; their presence in this distribution does not remove the need for content and rights review.

## Using or redistributing these materials

1. Obtain express permission for any intended LOL-v1/LOL-v2 and ExDark photographs and, where necessary, their metadata; or replace photographed examples with independently owned, original or clearly licensed images and rerun the visualizations with newly trained checkpoints.
2. For the report, obtain the rights to publish its exact photographs or prepare a separate public edition with authorised substitute images (explaining any substantial changes). The configured inclusion of the existing report PDF does not waive these requirements.
3. Preserve license and attribution information for MILL materials, if the relevant photographs are indeed covered by the stated MIT licence.
4. Check the terms of the CVPR style files or require installation from the official upstream kit before making a self-contained public TeX source package.
5. Before publication or redistribution, inspect the staged Git contents, review attribution and licences, and check for private information or secrets.
