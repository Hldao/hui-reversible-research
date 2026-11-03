"""Deterministic CPU wiring example on synthetic tokens, not a benchmark."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Optional, Union

import torch

from . import BACKENDS, build_model, parameter_count, tiny_config


def run_demo(
    backend: str = "exact", steps: int = 5, seed: int = 7,
    save: Optional[Union[str, Path]] = None,
    resume: Optional[Union[str, Path]] = None,
) -> dict:
    """Train until total ``steps``; resume only this example's own checkpoints.

    All computation uses CPU/FP32. Save paths must not already exist.
    Checkpoints contain model, optimizer, local data RNG and loss history.
    """
    if backend not in BACKENDS:
        raise ValueError("unsupported backend")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    destination = Path(save) if save is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError("checkpoint already exists: " + str(destination))
    config = tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = build_model(config, backend).to(device="cpu", dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = 0
    losses = []
    if resume is not None:
        state = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("schema") != "hui-tiny-demo-v1":
            raise ValueError("not a Hui tiny-demo checkpoint")
        if (state.get("backend") != backend or state.get("seed") != seed
                or state.get("config") != asdict(config)):
            raise ValueError("checkpoint backend, seed or model config does not match")
        start = state["completed_steps"]
        if type(start) is not int or not 0 < start <= steps:
            raise ValueError("target steps must be at least the checkpoint step")
        losses = state["losses"]
        if not isinstance(losses, list) or len(losses) != start:
            raise ValueError("checkpoint loss history does not match its step")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["data_rng"])
    for _ in range(start, steps):
        tokens = torch.randint(0, config.vocab_size, (2, 16), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, labels=tokens)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters()):
            raise FloatingPointError("missing or non-finite parameter gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("non-finite model parameter")
        losses.append(float(loss.detach()))
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            torch.save({
                "schema": "hui-tiny-demo-v1", "backend": backend, "seed": seed,
                "config": asdict(config), "completed_steps": steps,
                "losses": losses, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "data_rng": generator.get_state(),
            }, stream)
    return {
        "backend": backend, "completed_steps": steps, "final_loss": losses[-1],
        "losses": losses, "parameters": parameter_count(model),
        "device": "cpu", "dtype": "float32", "torch_version": str(torch.__version__),
        "synthetic_data": True, "historical_experiment_reproduction": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="exact")
    parser.add_argument("--steps", type=int, default=5, help="total target steps, including resumed steps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = run_demo(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
