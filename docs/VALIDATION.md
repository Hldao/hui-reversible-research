# Local candidate validation · 2026-11-03

Environment: Python 3.11, PyTorch 2.8.0, NumPy 2.2.6, macOS arm64, CPU. Dependencies installed into a new isolated virtual environment. No GPU experiments were executed.

## Verified

- 11 unittest cases passed against the installed wheel from outside the source checkout.
- All three installed CLI backends completed three synthetic training steps, with 31,430 parameters and final loss `4.174710273742676` in this smoke configuration. This equality is not a model-quality result.
- Deterministic save/resume was checked against uninterrupted training for model tensors, AdamW state, data-generator state and loss history.
- Tiny BF16 N2 gradients were finite and met the test's relative-error/cosine bounds. This is not validation of a 1.3B CUDA workload.
- Three extracted core modules match the frozen hashes after undoing the documented import-only transformations.
- Installed package copies match the packaged core hashes; source provenance is included in the wheel.
- Wheel built and installed; `pip check` reported no broken requirements.
- Wheel contents contain no model checkpoints, private absolute archive paths or matches for the checked common credential patterns. Pattern checks are not an exhaustive security or ownership review.

## Review correction

Read-only implementation review found N2 backward cannot accept frozen parameters inside reversible blocks. The adapter now rejects that training configuration before forward, with tests for rejection and inference. Frozen core files were not changed. The CPU demo also explicitly enforces FP32 without modifying a caller's global dtype.

## Not performed

No historical 50M rerun, CUDA benchmark, large-checkpoint content rehash, external-user installation, pretrained-checkpoint conversion, public release or legal/ownership clearance.

## Commands

```bash
python -m unittest discover -s /path/to/candidate/tests -v  # run outside the checkout after wheel install
python tools/verify_sources.py
python -m pip check
python -m hui.demo --backend exact --steps 3
python -m hui.demo --backend checkpoint --steps 3
python -m hui.demo --backend n2 --steps 3
```
