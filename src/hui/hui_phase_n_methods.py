#!/usr/bin/env python3
"""Frozen method dispatch helpers for the Hui Phase-N backward screen."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from .torch_module import ReversibleCausalBlock


METHODS = (
    "N0_EXACT",
    "N1_RAW_BF16",
    "N2_INT8_RESIDUAL",
    "N3_CHECKPOINT_K4",
    "N4_INT8_PLUS_K8",
)
SELECTABLE_METHODS = METHODS[1:]


def reversible_indices(model) -> tuple[int, ...]:
    return tuple(
        index
        for index, layer in enumerate(model.layers)
        if isinstance(layer, ReversibleCausalBlock)
    )


def periodic_boundaries(indices: tuple[int, ...], interval: int) -> tuple[int, ...]:
    """Return global layer indices at exact segment starts after each interval."""
    if interval < 1:
        raise ValueError("boundary interval must be positive")
    return tuple(indices[offset] for offset in range(interval, len(indices), interval))


@dataclass(frozen=True)
class MethodLayout:
    method_id: str
    reversible_indices: tuple[int, ...]
    exact_boundary_indices: tuple[int, ...]
    requested_risk_indices: tuple[int, ...]
    corrected_indices: tuple[int, ...]


def method_layout(model, method_id: str, risk_indices=()) -> MethodLayout:
    if method_id not in METHODS:
        raise ValueError(f"unknown Phase-N method: {method_id}")
    reversible = reversible_indices(model)
    reversible_set = set(reversible)
    requested_risk = tuple(sorted({int(value) for value in risk_indices}))
    if any(value not in reversible_set for value in requested_risk):
        raise ValueError("risk layers must be global reversible-layer indices")
    if method_id == "N3_CHECKPOINT_K4":
        boundaries = periodic_boundaries(reversible, 4)
    elif method_id == "N4_INT8_PLUS_K8":
        boundaries = periodic_boundaries(reversible, 8)
    else:
        boundaries = ()
    if method_id == "N2_INT8_RESIDUAL":
        corrected = reversible
    elif method_id == "N4_INT8_PLUS_K8":
        # An exact boundary dominates correction when the two sets overlap.
        corrected = tuple(value for value in requested_risk if value not in boundaries)
    else:
        corrected = ()
    return MethodLayout(
        method_id=method_id,
        reversible_indices=reversible,
        exact_boundary_indices=boundaries,
        requested_risk_indices=requested_risk,
        corrected_indices=corrected,
    )


@contextmanager
def apply_method(model, method_id: str, risk_indices=()):
    """Temporarily apply one Phase-N backward path without changing forward math."""
    layout = method_layout(model, method_id, risk_indices)
    layers = [model.layers[index] for index in layout.reversible_indices]
    original_flags = tuple(layer.memory_efficient for layer in layers)
    original_memory_efficient = model.config.memory_efficient_reversible
    original_checkpoints = tuple(model.config.checkpoint_reversible_layer_indices)
    original_corrected = tuple(model.config.corrected_reversible_layer_indices)
    original_segment_length = model.config.reversible_segment_length
    try:
        reconstruct = method_id != "N0_EXACT"
        model.config.memory_efficient_reversible = reconstruct
        model.config.reversible_segment_length = 0
        model.config.checkpoint_reversible_layer_indices = layout.exact_boundary_indices
        model.config.corrected_reversible_layer_indices = layout.corrected_indices
        for layer in layers:
            layer.memory_efficient = reconstruct
        yield layout
    finally:
        model.config.memory_efficient_reversible = original_memory_efficient
        model.config.checkpoint_reversible_layer_indices = original_checkpoints
        model.config.corrected_reversible_layer_indices = original_corrected
        model.config.reversible_segment_length = original_segment_length
        for layer, flag in zip(layers, original_flags):
            layer.memory_efficient = flag
