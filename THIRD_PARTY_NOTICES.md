# Third-party dependencies and attribution

This file records dependency and asset provenance. Hui selects Apache-2.0 in `LICENSE`; third-party assets retain their own terms. This inventory does not establish contributor permissions in the project's historical source.

## Runtime

**PyTorch** is installed separately; no PyTorch implementation or binary distribution is vendored in this candidate. The tested PyTorch 2.8.0 distribution identifies its project license as BSD-3-Clause and includes separate LICENSE and NOTICE files for its bundled components. Retain those distribution notices when redistributing PyTorch itself. Project: https://pytorch.org/ ; source license: https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE

The locally verified CPU environment additionally pins **NumPy 2.2.6** in `requirements-cpu.txt`. No NumPy binaries or source are included. Its distribution has its own license file and component notices. Project: https://numpy.org/ ; source license: https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt

## Building and regenerating manuscript figures

**Setuptools** is the separately installed build backend. **Matplotlib 3.10.7** is an optional figure-generation dependency in `paper/requirements.txt`; its software is not vendored. Its distribution provides the Matplotlib license and further asset notices. Project: https://matplotlib.org/ ; source license: https://github.com/matplotlib/matplotlib/blob/v3.10.7/LICENSE/LICENSE

The manuscript figures use **DejaVu Sans**. Figure PDFs may embed font subsets. The font notice copied without changes from the installed Matplotlib font assets is included at `third_party/DejaVu-LICENSE.txt`, covering the Bitstream Vera and Arev notices supplied with DejaVu. This font notice applies to those font assets, not Hui source code.

The manuscript PDF was generated using a separately downloaded Tectonic executable. Neither that executable nor a TeX installation is included. Its embedded font names are inventoried in `provenance/pdf-fonts.json`. Latin Modern notice texts are in `third_party/LatinModern/`; Computer Modern canonical Type1-family notices are in `third_party/AMSFonts/`. These were retrieved from CTAN package archives and the LaTeX Project. Font-family identification and supplied notice texts are recorded here; they do not grant rights in Hui itself. Sources: https://ctan.org/pkg/lm , https://ctan.org/pkg/amsfonts , https://www.latex-project.org/lppl/lppl-1-3c.txt .

## Historical accelerators

Some later archived experiments used Liger 0.8.2 fused linear cross-entropy and forced PyTorch Flash SDPA. Liger and external FlashAttention implementations are not part of this minimal source candidate. These historical trials do not establish compatibility of the extracted package with those dependencies. Refer to the paper for eager/compile and method distinctions.

## Scope of this review

This is an inventory of identified direct dependencies and generated assets, not a claim that every historical code fragment or transitive binary dependency has been legally cleared. Frozen-source hashes establish byte provenance; they do not establish ownership or permission. No dataset, checkpoint weight or historical Git history is included.
