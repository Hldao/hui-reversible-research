# Hui · Reversible Training Research

**Research pre-release · 0.1.0rc1 · Apache-2.0.** 中文说明见 [README.zh-CN.md](README.zh-CN.md).

Hui is a research implementation for memory-constrained causal-language-model training. It studies whether an INT8 correction for reversible residual blocks can reduce activation storage while keeping backward computations numerically close to a reference path.

The package lets you compare three backends on the same small model:

- `exact`: ordinary autograd, used as the numerical reference;
- `checkpoint`: full activation checkpointing, used as the practical resource baseline;
- `n2`: INT8 inverse-residual correction for reversible blocks, with checkpointing for standard blocks.

## What Hui is useful for

Hui is intended for researchers and engineers who want to:

- inspect how finite-precision reconstruction affects gradients and optimizer updates;
- compare reversible reconstruction with activation checkpointing under a fixed model and workload;
- prototype memory-constrained causal-LM training methods;
- reproduce the packaged CPU checks and study the audited historical GPU results.

It is a research instrument and a starting point for new backends. It is not a drop-in Llama/Qwen checkpoint converter, a production training framework, or a claim that N2 is faster or smaller in every setting.

## Evidence at a glance

The historical evidence and its boundaries are collected in [RESULTS.md](RESULTS.md), [the audited data](paper/data/audited-results.json), and [the A800 resource comparison](paper/data/a800-results.json).

| Question | What the current evidence shows |
|---|---|
| Does the correction improve reconstruction fidelity? | In the Phase N screen, N2 passed the registered direct-fidelity gates for 36/36 checkpoints; the uncorrected inverse path passed 24/36. |
| Does short training remain close to the reference? | Four paired 1.346B-parameter, 50M-token runs met the internal engineering tolerance. The mean validation-loss difference was -0.003759; four seeds are not enough to establish statistical non-inferiority. |
| Does N2 beat full checkpointing on resources? | Not in the matched A800 probe: at sequence length 4096, N2 reached about 0.717× the checkpoint baseline's throughput and used about 1.003× its allocated memory. |
| What is the positive memory result? | In the archived Phase D comparison, N2 reduced allocated memory by about 43.2% versus the saved-activation exact reference path. That comparison is not the same as full checkpointing. |

The numbers above come from different historical protocols. They are not one universal benchmark, and the current CPU demo does not reproduce the archived GPU runs.

## Possible application scenarios

- **Method research:** compare reversible reconstruction, checkpointing, and activation-precision choices.
- **Memory-constrained experiments:** explore whether a fixed model can run with a larger batch or sequence length under a defined memory budget.
- **Numerical debugging:** inspect gradient, AdamW-update, and save/resume behavior instead of treating final loss as the only signal.
- **Teaching and prototyping:** use the small CPU example to understand the training path before connecting a larger controlled experiment.

## Current boundaries

- The historical GPU experiments are not rerun by this public package.
- The CPU demo uses a tiny model and random tokens; it checks wiring and deterministic resume, not language quality or GPU memory savings.
- N2 training currently requires all parameters inside reversible blocks to remain trainable.
- General pretrained-checkpoint conversion and LoRA/PEFT are outside this candidate.
- The technical report is a working draft, not a peer-reviewed paper.

## Install and run

Python 3.11+ and PyTorch 2.8+ are required by this candidate. Local functional verification uses Python 3.11 and PyTorch 2.8.0 on macOS CPU. The historical 5090 experiments used a different pinned environment; see [evidence](docs/EVIDENCE.md).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.8.0' 'numpy==2.2.6'
python -m pip install --no-deps .
python -m hui.demo --backend exact --steps 5
python -m hui.demo --backend checkpoint --steps 5
python -m hui.demo --backend n2 --steps 5
```

The demo runs on CPU/FP32 and emits JSON with losses, step count, parameter count and environment. Synthetic random-token losses are a wiring check, not a language-model quality result. No dataset download, GPU job or background service is started.

Save and resume (steps is the total target, not additional steps):

```bash
python -m hui.demo --backend n2 --steps 2 --save partial.pt
python -m hui.demo --backend n2 --steps 5 --resume partial.pt --save resumed.pt
```

Existing save paths are never overwritten. Resume requires the same backend, seed and tiny-model configuration. It restores model, optimizer, local token-generator state and loss history. Only use checkpoints created by this example; historical 1.3B checkpoint loading is outside this demo's scope.

## Model API

```python
import torch
from hui import build_model, tiny_config

model = build_model(tiny_config(), backend="n2")
tokens = torch.randint(0, 64, (2, 16))
logits, loss = model(tokens, labels=tokens)
loss.backward()
```

`ExperimentConfig` is also exported for research configurations. Defaults in that historical dataclass describe a large model: start from `tiny_config()` for a local example. Keep vocabulary size, dimensions and token IDs consistent when supplying your own data. Reversible models require even full-model head counts and compatible branch widths.

| Backend | Behavior | Intended role |
|---|---|---|
| `exact` | Ordinary autograd, retaining required activations | Numerical reference; API default |
| `checkpoint` | Exact autograd with every layer checkpointed | Strong same-model resource baseline |
| `n2` | INT8 inverse-error correction on all reversible layers; standard layers checkpointed | Optional experimental reconstruction backend |

No backend is advertised as universally best. In archived matched tests, full checkpointing was faster and used slightly less allocated memory than N2. N2's positive evidence is finite-precision fidelity and passing engineering quality tolerances within a fixed historical configuration; four seeds do not establish statistical noninferiority.

The package preserves the historical forward architecture, including optional learned scales. Changing backend keeps the state-dict schema; the tiny resume interface deliberately does not certify backend switching.

N2 training currently requires every parameter inside a reversible block to remain trainable. Frozen parameters in those blocks are rejected before forward; use `exact` or `checkpoint` for partial freezing. Inference with frozen parameters is unaffected. LoRA/PEFT is not supported by this N2 candidate.

## Verify

```bash
python -m unittest discover -s tests -v
python tools/verify_sources.py
```

Tests cover FP32 forward/gradient comparisons, BF16 numerical fidelity on a tiny CPU model, actual INT8 saved corrections, finite parameter updates, and deterministic save/resume. These checks are not a re-run of the historical 1.3B/50M CUDA experiment or proof of GPU memory savings.

## Release and evidence

This is a research pre-release, not a production training framework. See [release notes](RELEASE_NOTES.md), [source provenance](docs/SOURCE_PROVENANCE.md), and [validation scope](docs/VALIDATION.md). Historical experiments and the current tiny CPU checks have different configurations; the tests do not reproduce the archived 1.3B training runs.

- [English technical report](paper/build/manuscript.pdf) (working draft, not peer reviewed)
- [Full Chinese report](paper/中文完整译稿.md)
- [Versioned release artifacts](https://github.com/Hldao/hui-reversible-research/releases/tag/v0.1.0-rc1)

Project/report attribution: **WKK Lab**. See [citation metadata](CITATION.cff), [license](LICENSE), [notices](NOTICE), and [third-party attribution](THIRD_PARTY_NOTICES.md).

This first release is a research snapshot. Issues and reproducible corrections are welcome; no response-time or ongoing GPU support commitment is made.
