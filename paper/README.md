# Hui manuscript draft

This is a publicly shared working technical-report draft by WKK Lab. It is not a peer-reviewed paper or an arXiv submission. Project-authored software and documentation use Apache-2.0; third-party assets retain their own notices.

- `build/manuscript.pdf`: English manuscript.
- `manuscript.tex`, `references.bib`: editable sources.
- `中文完整译稿.md`: full Chinese translation, including expanded appendices.
- `中文审阅稿.md`: shorter Chinese review version.
- `实验记录复核与完善建议.md`: record audit and prioritized improvements.
- `data/`: audited historical numeric summaries; no raw training checkpoints.
- `generated/`: regenerated tables and paired statistics.
- `figures/`: exportable PDF/PNG figures.
- `EVIDENCE.csv`: claim-to-record mapping and interpretation boundaries.

Regenerate figures with `python build_results.py` (Matplotlib 3.10.7 was used). Compile from the project root with `tectonic paper/manuscript.tex --outdir paper/build --keep-logs`, creating `paper/build` first. Tectonic 0.17.0 was used. These steps regenerate the manuscript, not the historical GPU experiments.

For any subsequent academic submission, confirm author and venue requirements and cite the frozen release; remaining scientific reproducibility items include complete historical hardware/software/config/command manifests, including per-run dtype, autocast and inverse-precision settings; accessible raw evidence subject to data permissions; and an independent, matched GPU reproduction. Four existing seed pairs do not establish statistical noninferiority at +0.005. No new training results were invented to close these gaps.

The expanded audit adds historical fixed-scale segmented C4/C8 task and two-stack records in separate appendices. These do not add N2 training seeds or resolve statistical noninferiority. Source-derived arithmetic provenance is recorded separately from incomplete runtime attestations.
