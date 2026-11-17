# Hui results and evidence map

This page is the short guide to what the public release can support. The full numeric records remain in [`paper/data/audited-results.json`](paper/data/audited-results.json), [`paper/data/a800-results.json`](paper/data/a800-results.json), and [`paper/data/extended-audit.json`](paper/data/extended-audit.json).

## Main evidence

### 1. Finite-precision reconstruction

The Phase N screen compared four strategies across 36 historical checkpoints and eight validation offsets per checkpoint. N2 passed the registered direct-fidelity gates for 36/36 checkpoints. The uncorrected inverse path passed 24/36.

The 1,152 checkpoint/offset/strategy records are repeated comparisons, not 1,152 independent training seeds.

### 2. Short matched training

The Phase D continuation used a 1.346B-parameter model, about 50M tokens, and four paired seeds.

- Mean N2-minus-exact validation-loss difference: **-0.003759**
- Internal engineering tolerance: **passed**
- Statistical non-inferiority at a +0.005 margin: **not established with four seeds**
- Later matched-window throughput ratio for N2 versus the N1 reference path: about **0.842**

The 43.2% memory figure belongs to this archived comparison against the saved-activation exact reference path. It must not be described as a 43.2% reduction versus full checkpointing.

### 3. Strong resource baseline

The A800 resource probe compared N2 with full exact checkpointing using eager PyTorch Flash SDPA.

At sequence length 4096:

| Backend | Tokens/s | Peak allocated memory |
|---|---:|---:|
| Full checkpoint | 12,102.8 | 11,795,025,408 bytes |
| N2 | 8,672.1 | 11,834,347,520 bytes |

That is approximately **0.717× throughput** and **1.003× allocated memory** for N2 relative to full checkpointing. This was a short resource probe, not a long training run, but it is enough to rule out a blanket claim that the current N2 implementation is the faster or smaller default.

## Later exploratory series

The later C4/C8/Liger, cross-backend, structured-task, and long-horizon records are useful for follow-up work. They use different fixed-scale recipes, workloads, devices, or single-seed protocols. They should remain labeled as exploratory appendices rather than being added to the N2 seed count.

The public package does not include historical checkpoints, private data, credentials, or a complete GPU reproduction environment. A new result should report the model, device, PyTorch/CUDA versions, dtype, sequence length, batch, backend, peak allocated and reserved memory, tokens/s, and quality metric.

## What a useful contribution looks like

A reproducible comparison against `exact`, `checkpoint`, or `n2` is more useful than a bare speed claim. Please include:

1. the command or runner configuration;
2. the hardware and software versions;
3. the sequence length, batch, seed, and number of updates;
4. peak allocated/reserved memory and tokens/s;
5. the quality or gradient-fidelity metric;
6. whether the result is a fresh run or a historical artifact.

Open an issue for a reproducible correction or a clearly scoped new comparison.
