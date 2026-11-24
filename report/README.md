# University project report — PDF and LaTeX source

This 45-page university report documents the original low-light enhancement project. It was composed using the **CVPR 2026 LaTeX author-kit style**, but it is **not** an accepted CVPR publication.

The repository includes the 45-page report PDF, complete LaTeX text, bibliography, nine figures with their `*.tex` inclusion snippets, `cvpr.sty`, and `ieeenat_fullname.bst`. The distribution includes all of these inputs but omits generated LaTeX compiler by-products. The report contains photographic panels derived from third-party datasets, whose redistribution rights must be reviewed before public release.

## Build

From the `report/` directory, with a full TeX Live installation including `pdflatex`, `bibtex` and the needed packages:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Compiler outputs are generated locally; do not commit temporary compilation files.

Build requirements include a working `latexmk` or `pdflatex`/`bibtex` installation and the LaTeX packages used by `preamble.tex`. The PDF distributed here was generated from the report source; rebuilding on a different TeX installation may produce minor formatting differences.

## Third-party rights and publication

Six photo-containing result figures (`ecl_far_intermediate`, `ecl_far_qualitative`, `fac_ecl_qualitative`, `illumination_guided_unet_qualitative`, `inter_family_random_panel`, `unet_qualitative`) reproduce or transform images from LOL-v1, LOL-v2, MILLs, or ExDark. A project-authored panel **does not itself establish a redistribution licence over the source photographs**. The PDF embeds these figures and is therefore not automatically cleared for public posting. `docs/publication-rights.md` explains the source-specific redistribution considerations.

The `cvpr.sty` and `ieeenat_fullname.bst` files originate from the [official CVPR/ICCV/3DV LaTeX author kit](https://github.com/cvpr-org/author-kit); do not misrepresent them as original code. Preserve upstream attribution and consult the author-kit terms when redistributing these files.

The repository configuration now includes the complete report and figures. **This does not establish that the embedded photographs are cleared for redistribution.** Obtain the necessary permissions or confirm an applicable exception before publishing the PDF and photo-based panels publicly; see `docs/publication-rights.md`.
