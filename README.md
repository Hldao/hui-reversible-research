# Hui · Reversible Training Research

**Research pre-release · 0.1.0rc1 · Apache-2.0.** 中文说明见 [README.zh-CN.md](README.zh-CN.md).

Hui is a research implementation of a hybrid reversible causal language model. It includes learned residual scales, three backward choices, and the N2 INT8 inverse-residual correction. The core comes from a hash-verified historical experiment snapshot; see [source provenance](provenance/sources.json).

This candidate is for researchers exploring training under memory constraints. It contains a small synthetic example that runs without datasets or pretrained weights. It does not convert existing Llama/Qwen checkpoints or establish production performance.

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
