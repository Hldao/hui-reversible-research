"""Decoder-only language models for the DeepSeek-Coder-1.3B controlled A/B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .torch_module import (
    CausalSelfAttention,
    RMSNorm,
    ReversibleCausalBlock,
    reversible_block_with_int8_correction,
    reversible_sequence,
)


class FixedSwiGLU(nn.Module):
    def __init__(self, width: int, inner: int, chunk_size: int = 0) -> None:
        super().__init__()
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)
        self.chunk_size = chunk_size

    def forward(self, x):
        if self.chunk_size and x.shape[-2] > self.chunk_size:
            outputs = []
            for start in range(0, x.shape[-2], self.chunk_size):
                chunk = x[:, start : start + self.chunk_size]
                if self.training and torch.is_grad_enabled():
                    output = checkpoint(
                        self._forward_dense, chunk, use_reentrant=False
                    )
                else:
                    output = self._forward_dense(chunk)
                outputs.append(output)
            return torch.cat(outputs, dim=-2)
        return self._forward_dense(x)

    def _forward_dense(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StandardDecoderBlock(nn.Module):
    def __init__(self, config: "ExperimentConfig") -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            rope_theta=config.rope_theta,
            rope_theta_global=config.rope_theta_global,
            rope_global_head_fraction=config.rope_global_head_fraction,
            attention_window=config.attention_window,
            attention_sink_tokens=config.attention_sink_tokens,
            attention_global_stride=config.attention_global_stride,
        )
        self.ffn = FixedSwiGLU(
            config.hidden_size,
            config.intermediate_size,
            chunk_size=config.ffn_chunk_size,
        )
        learn_standard_scale = not config.learnable_reversible_scale_only
        if config.bounded_residual_scale and learn_standard_scale:
            fraction = (
                config.residual_scale_init - config.residual_scale_min
            ) / (config.residual_scale_max - config.residual_scale_min)
            logit = torch.tensor(math.log(fraction / (1.0 - fraction)))
            self.attention_scale_logit = nn.Parameter(logit.clone())
            self.ffn_scale_logit = nn.Parameter(logit.clone())
        else:
            scale = torch.tensor(float(config.residual_scale_init))
        if (
            config.learnable_residual_scale
            and not config.bounded_residual_scale
            and learn_standard_scale
        ):
            self.attention_scale = nn.Parameter(scale.clone())
            self.ffn_scale = nn.Parameter(scale.clone())
        elif not (config.bounded_residual_scale and learn_standard_scale):
            self.register_buffer("attention_scale", scale.clone())
            self.register_buffer("ffn_scale", scale.clone())
        self.residual_scale_min = config.residual_scale_min
        self.residual_scale_max = config.residual_scale_max

    def forward(self, x):
        x = x + self._scale("attention") * self.attention(
            self.attention_norm(x)
        )
        return x + self._scale("ffn") * self.ffn(self.ffn_norm(x))

    def _scale(self, branch):
        logit = getattr(self, f"{branch}_scale_logit", None)
        if logit is None:
            return getattr(self, f"{branch}_scale")
        return self.residual_scale_min + (
            self.residual_scale_max - self.residual_scale_min
        ) * logit.sigmoid()


@dataclass
class ExperimentConfig:
    vocab_size: int = 32256
    hidden_size: int = 2048
    intermediate_size: int = 5504
    reversible_intermediate_size: int = 15104
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 16384
    rope_theta: float = 100000.0
    rope_theta_global: Optional[float] = None
    rope_global_head_fraction: float = 0.0
    rms_norm_eps: float = 1e-6
    architecture: str = "standard"
    activation_checkpointing: bool = False
    activation_checkpoint_fraction: float = 1.0
    checkpoint_standard_layers_only: bool = False
    memory_efficient_reversible: bool = False
    reversible_segment_length: int = 0
    late_reversible_segment_length: int = 0
    late_reversible_segment_start: int = 0
    short_reversible_segment_length: int = 0
    short_reversible_segment_starts: tuple[int, ...] = ()
    checkpoint_reversible_layer_indices: tuple[int, ...] = ()
    corrected_reversible_layer_indices: tuple[int, ...] = ()
    learnable_residual_scale: bool = False
    learnable_reversible_scale_only: bool = False
    residual_scale_init: float = 1.0
    bounded_residual_scale: bool = False
    residual_scale_min: float = 0.1
    residual_scale_max: float = 1.0
    reversible_channel_mixing: str = "alternating_swap"
    reversible_swap_period: int = 2
    reversible_fraction: float = 0.5
    non_swap_attention_scale_init: Optional[float] = None
    non_swap_ffn_scale_init: Optional[float] = None
    swap_attention_scale_init: Optional[float] = None
    swap_ffn_scale_init: Optional[float] = None
    fp32_inverse_accumulation: bool = False
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    loss_chunk_size: int = 0
    attention_window: Optional[int] = None
    attention_sink_tokens: int = 0
    attention_global_stride: int = 0
    ffn_chunk_size: int = 0


class ControlledCausalLM(nn.Module):
    """Matched language-model shell; only the decoder block family changes."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        if not 0.0 <= config.activation_checkpoint_fraction <= 1.0:
            raise ValueError(
                "activation_checkpoint_fraction must be in [0, 1]"
            )
        if config.reversible_segment_length < 0:
            raise ValueError("reversible_segment_length must be non-negative")
        if config.late_reversible_segment_length < 0:
            raise ValueError(
                "late_reversible_segment_length must be non-negative"
            )
        if config.late_reversible_segment_start < 0:
            raise ValueError(
                "late_reversible_segment_start must be non-negative"
            )
        if config.short_reversible_segment_length < 0:
            raise ValueError(
                "short_reversible_segment_length must be non-negative"
            )
        if any(index < 0 for index in config.short_reversible_segment_starts):
            raise ValueError(
                "short_reversible_segment_starts must be non-negative"
            )
        special_indices = (
            set(config.checkpoint_reversible_layer_indices)
            | set(config.corrected_reversible_layer_indices)
        )
        if any(index < 0 for index in special_indices):
            raise ValueError("special reversible layer indices must be non-negative")
        if (
            set(config.checkpoint_reversible_layer_indices)
            & set(config.corrected_reversible_layer_indices)
        ):
            raise ValueError("a reversible layer cannot use two fallback strategies")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.architecture == "standard":
            self.layers = nn.ModuleList(
                StandardDecoderBlock(config)
                for _ in range(config.num_hidden_layers)
            )
        elif config.architecture in ("reversible", "hybrid"):
            if config.num_attention_heads % 2:
                raise ValueError(
                    "reversible model needs an even full-model head count"
                )
            layers = []
            if not 0.0 < config.reversible_fraction <= 1.0:
                raise ValueError("reversible_fraction must be in (0, 1]")
            reversible_count = (
                config.num_hidden_layers
                if config.architecture == "reversible"
                else round(
                    config.num_hidden_layers * config.reversible_fraction
                )
            )
            for index in range(config.num_hidden_layers):
                use_reversible = (
                    config.architecture == "reversible"
                    or (
                        (index + 1)
                        * reversible_count
                        // config.num_hidden_layers
                        > index
                        * reversible_count
                        // config.num_hidden_layers
                    )
                )
                if use_reversible:
                    if config.reversible_channel_mixing == "periodic_swap":
                        if config.reversible_swap_period < 1:
                            raise ValueError(
                                "reversible_swap_period must be positive"
                            )
                        output_mixing = (
                            "swap"
                            if (index + 1) % config.reversible_swap_period == 0
                            else "none"
                        )
                    elif config.reversible_channel_mixing in (
                        "alternating_swap",
                        "alternating_hadamard",
                    ):
                        mixed = config.reversible_channel_mixing.removeprefix(
                            "alternating_"
                        )
                        output_mixing = mixed if index % 2 else "none"
                    elif config.reversible_channel_mixing in (
                        "none",
                        "hadamard",
                    ):
                        output_mixing = config.reversible_channel_mixing
                    else:
                        raise ValueError(
                            "reversible_channel_mixing must be "
                            "periodic_swap, alternating_swap, "
                            "alternating_hadamard, "
                            "none, or hadamard"
                        )
                    layers.append(
                        ReversibleCausalBlock(
                            config.hidden_size,
                            config.num_attention_heads // 2,
                            ffn_inner_size=config.reversible_intermediate_size,
                            output_mixing=output_mixing,
                            memory_efficient=(
                                config.memory_efficient_reversible
                                and config.reversible_segment_length <= 1
                            ),
                            rope_theta=config.rope_theta,
                            rope_theta_global=config.rope_theta_global,
                            rope_global_head_fraction=(
                                config.rope_global_head_fraction
                            ),
                            learnable_residual_scale=(
                                config.learnable_residual_scale
                            ),
                            residual_scale_init=config.residual_scale_init,
                            attention_scale_init=(
                                config.swap_attention_scale_init
                                if output_mixing == "swap"
                                else config.non_swap_attention_scale_init
                            ),
                            ffn_scale_init=(
                                config.swap_ffn_scale_init
                                if output_mixing == "swap"
                                else config.non_swap_ffn_scale_init
                            ),
                            bounded_residual_scale=(
                                config.bounded_residual_scale
                            ),
                            residual_scale_min=config.residual_scale_min,
                            residual_scale_max=config.residual_scale_max,
                            fp32_inverse_accumulation=(
                                config.fp32_inverse_accumulation
                            ),
                            attention_window=config.attention_window,
                            attention_sink_tokens=(
                                config.attention_sink_tokens
                            ),
                            attention_global_stride=(
                                config.attention_global_stride
                            ),
                            ffn_chunk_size=config.ffn_chunk_size,
                        )
                    )
                else:
                    layers.append(StandardDecoderBlock(config))
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError(
                "architecture must be 'standard', 'hybrid', or 'reversible'"
            )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def reversible_segments(self) -> list[list[int]]:
        """Return the exact activation-reconstruction segments used in backward."""
        special_indices = (
            set(self.config.checkpoint_reversible_layer_indices)
            | set(self.config.corrected_reversible_layer_indices)
        )
        default_length = self.config.reversible_segment_length
        if default_length <= 0:
            default_length = 1
        segments = []
        reversible_ordinal = 0
        index = 0
        while index < len(self.layers):
            if not isinstance(self.layers[index], ReversibleCausalBlock):
                index += 1
                continue
            if index in special_indices:
                index += 1
                reversible_ordinal += 1
                continue
            length = default_length
            if (
                self.config.short_reversible_segment_length > 0
                and reversible_ordinal
                in self.config.short_reversible_segment_starts
            ):
                length = self.config.short_reversible_segment_length
            if (
                self.config.late_reversible_segment_length > 0
                and reversible_ordinal
                >= self.config.late_reversible_segment_start
            ):
                length = self.config.late_reversible_segment_length
            segment = []
            while (
                index < len(self.layers)
                and len(segment) < length
                and isinstance(self.layers[index], ReversibleCausalBlock)
                and index not in special_indices
            ):
                segment.append(index)
                index += 1
                reversible_ordinal += 1
            segments.append(segment)
        return segments

    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        labels: Optional[torch.Tensor] = None,
        logits_positions: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        hidden = self.embed_tokens(input_ids)
        checkpoint_count = round(
            len(self.layers) * self.config.activation_checkpoint_fraction
        )
        segment_stops = {
            segment[0]: segment[-1] + 1
            for segment in self.reversible_segments()
        }
        index = 0
        while index < len(self.layers):
            layer = self.layers[index]
            if (
                self.training
                and torch.is_grad_enabled()
                and index in self.config.checkpoint_reversible_layer_indices
            ):
                hidden = checkpoint(layer, hidden, use_reentrant=False)
                index += 1
                continue
            if (
                self.training
                and torch.is_grad_enabled()
                and index in self.config.corrected_reversible_layer_indices
            ):
                hidden = reversible_block_with_int8_correction(hidden, layer)
                index += 1
                continue
            if (
                self.training
                and torch.is_grad_enabled()
                and self.config.memory_efficient_reversible
                and self.config.reversible_segment_length > 0
                and isinstance(layer, ReversibleCausalBlock)
            ):
                stop = segment_stops[index]
                hidden = reversible_sequence(hidden, self.layers[index:stop])
                index = stop
                continue
            checkpoint_layer = (
                (index + 1) * checkpoint_count // len(self.layers)
                > index * checkpoint_count // len(self.layers)
            )
            if (
                self.config.activation_checkpointing
                and self.training
                and checkpoint_layer
                and (
                    not self.config.checkpoint_standard_layers_only
                    or not isinstance(layer, ReversibleCausalBlock)
                )
            ):
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
            index += 1
        hidden = self.norm(hidden)
        if logits_positions is not None:
            if labels is not None:
                raise ValueError(
                    "labels and logits_positions cannot be combined"
                )
            if logits_positions.ndim != 2:
                raise ValueError(
                    "logits_positions must have shape [batch, positions]"
                )
            if logits_positions.shape[0] != hidden.shape[0]:
                raise ValueError(
                    "logits_positions batch dimension must match input"
                )
            gather_index = logits_positions[..., None].expand(
                -1, -1, hidden.shape[-1]
            )
            selected = hidden.gather(1, gather_index)
            if return_hidden:
                return selected
            return self.lm_head(selected)
        if return_hidden:
            if labels is not None:
                raise ValueError("labels and return_hidden cannot be combined")
            return hidden
        if labels is not None and self.config.loss_chunk_size:
            if self.config.loss_chunk_size < 1:
                raise ValueError("loss_chunk_size must be non-negative")
            prediction_hidden = hidden[:, :-1]
            targets = labels[:, 1:]
            weighted_losses = []
            token_counts = []
            for start in range(
                0, prediction_hidden.shape[1], self.config.loss_chunk_size
            ):
                end = min(
                    start + self.config.loss_chunk_size,
                    prediction_hidden.shape[1],
                )
                hidden_chunk = prediction_hidden[:, start:end]
                target_chunk = targets[:, start:end]

                def chunk_loss(chunk, chunk_targets):
                    chunk_logits = self.lm_head(chunk)
                    return F.cross_entropy(
                        chunk_logits.float().reshape(
                            -1, chunk_logits.size(-1)
                        ),
                        chunk_targets.reshape(-1),
                    )

                if self.training and torch.is_grad_enabled():
                    loss_part = checkpoint(
                        chunk_loss,
                        hidden_chunk,
                        target_chunk,
                        use_reentrant=False,
                    )
                else:
                    loss_part = chunk_loss(hidden_chunk, target_chunk)
                count = target_chunk.numel()
                weighted_losses.append(loss_part * count)
                token_counts.append(count)
            loss = sum(weighted_losses) / sum(token_counts)
            return None, loss

        logits = self.lm_head(hidden)
        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().float().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        return logits, loss


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
