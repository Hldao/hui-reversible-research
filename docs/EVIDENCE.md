# Evidence and interpretation boundaries

The following historical results are evidence references, not results produced by this candidate's tiny demo. Archive availability and numerical claims were checked during the 2026-11-03 audit; large checkpoint contents were not rehashed in that audit.

| Evidence | Scope | Appropriate conclusion |
|---|---|---|
| Partner-side `124m-phase1`, commit `b68f29e06f7d9807d4ababec32a6f16d423c8d00` | Two recorded GPU identities; B/C/D/H; 8 runs; 8,192 tokens per run; metrics hashes verified | Historical clean-checkout short-run reproducibility; not N2/50M replication |
| Phase N, 2026-08-08 | 36 exact checkpoints, 8 offsets, four candidates | N2 was the only candidate passing all registered direct fidelity gates |
| Phase D, 2026-08-08 | 1.3B, seq512, batch8, 50M tokens, four seeds, RTX 5090 | N2 passed internal engineering quality tolerances and numerical gates; four seeds do not establish statistical noninferiority at +0.005 |
| Phase D matched throughput adjudication | Paired N1/N2 timing windows | N2 retained about 84.2% of raw-reconstruction throughput; this is not a comparison with full checkpointing |
| K1/N2 context ladder, 2026-08-09 | Same-model 512/2K/4K resource probes and four-seed 4K short continuation | Full exact checkpointing slightly beat K1 on speed/memory; K1 beat N2 on speed. K1 did not pass every 4K continuation quality gate |
| A800 matched Flash SDPA eager probes, 2026-08-20 | 512 calibration and 4K primary resource comparison, three trials per method | N2/full throughput about 0.713/0.717, allocated-memory ratio about 1.0037/1.0033; no N2 resource advantage |

Phase D's roughly 43.2% allocated-memory reduction compares with saved-activation exact backward of the same model. It does not establish a 43.2% reduction relative to optimized full checkpointing. Allocated memory alone is not the minimum GPU capacity: reserved, workspaces and other allocations also matter.

The original throughput failure and subsequent matched-window adjudication are separate preserved records. Historical N2 50M reproduction by a second executor has not been located in the searched legacy partner-node mirrors; this is an unresolved evidence location, not a claim that it never happened.

The packaged core is tied to Phase N/D by `provenance/sources.json`. The newly written backend interface and demo have their own functional tests. Import-only extraction preserves source mathematics, but a different PyTorch version, hardware or workload can change numerical behavior and resource results.

No universal throughput advantage, effective 128K memory, general pretrained-model conversion or production readiness is claimed.

The expanded manuscript audit distinguishes later fixed-scale segmented C4/C8 variants and task/cross-backend canaries from the main N2 method. See `paper/data/extended-audit.json` and the manuscript appendices. These later records are not extra N2 seeds or independent N2 replications.
