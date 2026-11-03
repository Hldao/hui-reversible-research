"""Explicit backward choices without modifying the frozen forward model."""

from dataclasses import replace

import torch

from .language_model import ControlledCausalLM, ExperimentConfig
from .torch_module import ReversibleCausalBlock

BACKENDS = ("exact", "checkpoint", "n2")


def _require_trainable_reversible_parameters(model, inputs):
    if not model.training or not torch.is_grad_enabled():
        return
    for layer in model.layers:
        if isinstance(layer, ReversibleCausalBlock):
            if any(not parameter.requires_grad for parameter in layer.parameters()):
                raise ValueError(
                    "n2 requires all reversible-block parameters to remain trainable; "
                    "use exact or checkpoint for frozen parameters"
                )


def tiny_config() -> ExperimentConfig:
    """Small synthetic example, not a scaled reproduction of Phase D."""
    return ExperimentConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        reversible_intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, max_position_embeddings=128,
        architecture="hybrid", reversible_fraction=0.75,
        residual_scale_init=0.25, learnable_residual_scale=True,
        learnable_reversible_scale_only=True, bounded_residual_scale=True,
        residual_scale_min=0.15, residual_scale_max=0.5,
        fp32_inverse_accumulation=True,
    )


def build_model(config: ExperimentConfig, backend: str = "exact") -> ControlledCausalLM:
    """Build a fresh model; backend choices preserve its state-dict schema.

    ``checkpoint`` checkpoints every layer with exact autograd. ``n2``
    corrects every reversible layer and checkpoints the standard layers.
    Existing legacy segment/fallback flags are cleared in a copied config.
    """
    if backend not in BACKENDS:
        raise ValueError("backend must be exact, checkpoint, or n2")
    for name in ("vocab_size", "hidden_size", "intermediate_size",
                 "reversible_intermediate_size", "num_hidden_layers", "num_attention_heads"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config.architecture not in ("standard", "hybrid", "reversible"):
        raise ValueError("unsupported model architecture")
    if backend == "n2" and config.architecture == "standard":
        raise ValueError("n2 requires reversible layers")
    clean = replace(
        config, activation_checkpointing=backend != "exact",
        activation_checkpoint_fraction=1.0,
        checkpoint_standard_layers_only=backend == "n2",
        memory_efficient_reversible=False, reversible_segment_length=0,
        late_reversible_segment_length=0, late_reversible_segment_start=0,
        short_reversible_segment_length=0, short_reversible_segment_starts=(),
        checkpoint_reversible_layer_indices=(), corrected_reversible_layer_indices=(),
    )
    model = ControlledCausalLM(clean)
    if backend == "n2":
        indices = tuple(i for i, layer in enumerate(model.layers)
                        if isinstance(layer, ReversibleCausalBlock))
        if not indices:
            raise ValueError("n2 requires at least one reversible layer")
        model.config.corrected_reversible_layer_indices = indices
        model.register_forward_pre_hook(_require_trainable_reversible_parameters)
    return model
