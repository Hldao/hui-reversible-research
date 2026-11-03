# Hui 0.1.0rc1 — research pre-release

Version: 0.1.0rc1. License: Apache-2.0. Project/report attribution: WKK Lab.

Hui provides a research implementation of hybrid reversible causal-language-model training. WKK Lab provides a research snapshot of the implementation, documentation and experimental boundaries; the selected project terms are in `LICENSE`, with attribution information in `NOTICE`.

## Included

- Three hash-tracked frozen historical modules; package-relative imports are the only core transformations.
- `exact`, `checkpoint` and `n2` adapters for a common forward architecture.
- Small CPU/FP32 synthetic training and deterministic checkpoint/resume demonstration.
- Tests, source verification, bilingual documentation, and citation metadata.
- A technical-report draft and audited numeric summaries separating main N2 results from later segmented variants.

## What the evidence establishes

Residual correction improves observed backward fidelity in the documented scope. Four paired approximately 50M-token runs satisfy engineering quality tolerances, but statistical noninferiority at +0.005 is not established. Full checkpointing is a strong baseline and wins the main matched short resource comparison. Later Liger-backed segmented variants use different recipes and workloads.

## Limitations

No universal speedup or depth-independent activation memory is claimed. The CPU demonstration does not reproduce historical GPU quality runs. N2 does not support frozen parameters inside reversible blocks during gradient-enabled training. No PEFT/LoRA, general pretrained-model conversion, production readiness or independent N2 50M replication is claimed.

No weights, corpus, deployment configuration, credentials or historical Git history are included. Repository: https://github.com/Hldao/hui-reversible-research . This technical report is a working draft, not a peer-reviewed publication. No DOI or arXiv identifier is claimed.
