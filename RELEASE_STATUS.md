# Research pre-release — 0.1.0rc1

License: Apache-2.0. Project/report attribution: WKK Lab. Git tag: `v0.1.0-rc1`.

Included: frozen core with source hashes, exact/checkpoint/N2 adapters, synthetic CPU training and deterministic save/resume, numerical tests, bilingual documentation, and a working technical report with audited summaries and identified font notices.

The release intentionally excludes weights, training corpus, private infrastructure, historical Git history and later C4/C8 execution implementations. Later experiment summaries do not imply those paths are supported by this package.

Validation is scoped to the documented CPU environment and installed-package checks. Historical GPU training is not rerun. Full checkpointing is a strong baseline; N2's four seed pairs do not establish statistical noninferiority. See `docs/VALIDATION.md`, `docs/EVIDENCE.md`, and `docs/SOURCE_PROVENANCE.md`.

This is a research snapshot; no production-readiness or ongoing support commitment is made.
