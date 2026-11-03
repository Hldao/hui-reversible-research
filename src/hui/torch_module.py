"""PyTorch module for integration into a decoder-only training repository."""

from __future__ import annotations

from typing import Optional
import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
except ImportError as exc:  # Make the missing optional dependency explicit.
    raise ImportError("torch_module.py requires PyTorch in the training environment") from exc


_COMPILED_FLEX_ATTENTION = None
_FLEX_BLOCK_MASK_CACHE = {}


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention, dynamic=False
        )
    return _COMPILED_FLEX_ATTENTION


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        bias: bool = False,
        rope_theta: float = 10000.0,
        rope_theta_global: Optional[float] = None,
        rope_global_head_fraction: float = 0.0,
        attention_window: Optional[int] = None,
        attention_sink_tokens: int = 0,
        attention_global_stride: int = 0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("branch width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.rope_theta = rope_theta
        self.rope_theta_global = rope_theta_global
        self.rope_global_head_fraction = rope_global_head_fraction
        if attention_window is not None and attention_window < 1:
            raise ValueError("attention_window must be positive")
        if attention_sink_tokens < 0:
            raise ValueError("attention_sink_tokens must be non-negative")
        if attention_global_stride < 0:
            raise ValueError("attention_global_stride must be non-negative")
        self.attention_window = attention_window
        self.attention_sink_tokens = attention_sink_tokens
        self.attention_global_stride = attention_global_stride
        self.qkv = nn.Linear(width, width * 3, bias=bias)
        self.output = nn.Linear(width, width, bias=bias)

    def _apply_rope(self, x):
        sequence = x.shape[-2]
        dimensions = (
            torch.arange(0, self.head_dim, 2, device=x.device).float()
            / self.head_dim
        )
        thetas = torch.full(
            (self.heads,),
            float(self.rope_theta),
            device=x.device,
        )
        global_heads = round(self.heads * self.rope_global_head_fraction)
        if self.rope_theta_global is not None and global_heads:
            thetas[-global_heads:] = float(self.rope_theta_global)
        frequency = 1.0 / thetas[:, None].pow(dimensions[None, :])
        positions = torch.arange(sequence, device=x.device).float()
        angles = positions[:, None, None] * frequency[None, :, :]
        cos = angles.cos().to(x.dtype).permute(1, 0, 2)[None, :, :, :]
        sin = angles.sin().to(x.dtype).permute(1, 0, 2)[None, :, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def forward(self, x):
        batch, sequence, width = x.shape
        qkv = self.qkv(x).view(
            batch, sequence, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (
            value.transpose(1, 2).contiguous() for value in (q, k, v))
        q, k = self._apply_rope(q), self._apply_rope(k)
        if self.attention_window is None:
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True
            )
        else:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
            )

            cache_key = (
                sequence,
                str(q.device),
                self.attention_window,
                self.attention_sink_tokens,
                self.attention_global_stride,
            )
            block_mask = _FLEX_BLOCK_MASK_CACHE.get(cache_key)
            if block_mask is None:
                window = self.attention_window
                sinks = self.attention_sink_tokens
                global_stride = self.attention_global_stride

                def causal_window_mask(batch, head, query_index, key_index):
                    causal = key_index <= query_index
                    local = query_index - key_index < window
                    sink = key_index < sinks
                    relay = (
                        global_stride > 0
                        and (key_index + 1) % global_stride == 0
                    )
                    return causal & (local | sink | relay)

                block_mask = create_block_mask(
                    causal_window_mask,
                    B=None,
                    H=None,
                    Q_LEN=sequence,
                    KV_LEN=sequence,
                    device=q.device,
                    _compile=True,
                )
                _FLEX_BLOCK_MASK_CACHE[cache_key] = block_mask
            attended = _compiled_flex_attention()(
                q, k, v, block_mask=block_mask
            )
        return self.output(
            attended.transpose(1, 2).reshape(batch, sequence, width))


class SwiGLU(nn.Module):
    def __init__(
        self,
        width: int,
        expansion: float = 8 / 3,
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        inner = max(8, int(width * expansion))
        inner = ((inner + 7) // 8) * 8
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


class ReversibleCausalBlock(nn.Module):
    """Algebraically reversible causal decoder block.

    Set ``memory_efficient=True`` to discard the block input during forward.
    Backward reconstructs the input from the output and recomputes the block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        ffn_expansion: float = 8 / 3,
        ffn_inner_size: Optional[int] = None,
        swap_output: bool = False,
        output_mixing: Optional[str] = None,
        memory_efficient: bool = False,
        rope_theta: float = 10000.0,
        rope_theta_global: Optional[float] = None,
        rope_global_head_fraction: float = 0.0,
        learnable_residual_scale: bool = False,
        residual_scale_init: float = 1.0,
        attention_scale_init: Optional[float] = None,
        ffn_scale_init: Optional[float] = None,
        bounded_residual_scale: bool = False,
        residual_scale_min: float = 0.1,
        residual_scale_max: float = 1.0,
        fp32_inverse_accumulation: bool = False,
        attention_window: Optional[int] = None,
        attention_sink_tokens: int = 0,
        attention_global_stride: int = 0,
        ffn_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even")
        branch = hidden_size // 2
        self.attention_norm = RMSNorm(branch)
        self.ffn_norm = RMSNorm(branch)
        self.attention = CausalSelfAttention(
            branch,
            num_heads,
            rope_theta=rope_theta,
            rope_theta_global=rope_theta_global,
            rope_global_head_fraction=rope_global_head_fraction,
            attention_window=attention_window,
            attention_sink_tokens=attention_sink_tokens,
            attention_global_stride=attention_global_stride,
        )
        effective_expansion = (
            ffn_inner_size / branch
            if ffn_inner_size is not None
            else ffn_expansion
        )
        self.ffn = SwiGLU(
            branch, effective_expansion, chunk_size=ffn_chunk_size
        )
        self.output_mixing = output_mixing or (
            "swap" if swap_output else "none"
        )
        if self.output_mixing not in ("none", "swap", "hadamard"):
            raise ValueError("output_mixing must be none, swap, or hadamard")
        self.swap_output = self.output_mixing == "swap"
        self.memory_efficient = memory_efficient
        self.fp32_inverse_accumulation = fp32_inverse_accumulation
        attention_init = (
            residual_scale_init
            if attention_scale_init is None
            else attention_scale_init
        )
        ffn_init = (
            residual_scale_init if ffn_scale_init is None else ffn_scale_init
        )
        if bounded_residual_scale:
            if not (
                residual_scale_min < attention_init < residual_scale_max
                and residual_scale_min < ffn_init < residual_scale_max
            ):
                raise ValueError("bounded scale init must lie inside its bounds")
            attention_fraction = (attention_init - residual_scale_min) / (
                residual_scale_max - residual_scale_min
            )
            ffn_fraction = (ffn_init - residual_scale_min) / (
                residual_scale_max - residual_scale_min
            )
            self.attention_scale_logit = nn.Parameter(
                torch.tensor(
                    math.log(
                        attention_fraction / (1.0 - attention_fraction)
                    )
                )
            )
            self.ffn_scale_logit = nn.Parameter(
                torch.tensor(
                    math.log(ffn_fraction / (1.0 - ffn_fraction))
                )
            )
            self.residual_scale_min = residual_scale_min
            self.residual_scale_max = residual_scale_max
        else:
            attention_scale = torch.tensor(float(attention_init))
            ffn_scale = torch.tensor(float(ffn_init))
        if learnable_residual_scale and not bounded_residual_scale:
            self.attention_scale = nn.Parameter(attention_scale)
            self.ffn_scale = nn.Parameter(ffn_scale)
        elif not bounded_residual_scale:
            self.register_buffer("attention_scale", attention_scale)
            self.register_buffer("ffn_scale", ffn_scale)

    def forward(self, x):
        if self.memory_efficient and torch.is_grad_enabled():
            return _ReversibleBlockFunction.apply(
                x, self, *tuple(self.parameters()))
        return self._forward_impl(x)

    def _forward_impl(self, x):
        xa, xb = x.chunk(2, dim=-1)
        yb = xb + self._scale("attention") * self.attention(
            self.attention_norm(xa)
        )
        ya = xa + self._scale("ffn") * self.ffn(self.ffn_norm(yb))
        return self._mix_output(ya, yb)

    def inverse(self, y):
        ya, yb = self._unmix_output(y)
        ffn_update = self.ffn(self.ffn_norm(yb))
        if self.fp32_inverse_accumulation and y.dtype != torch.float32:
            xa = (
                ya.float()
                - self._scale("ffn").float() * ffn_update.float()
            ).to(y.dtype)
        else:
            xa = ya - self._scale("ffn") * ffn_update
        attention_update = self.attention(self.attention_norm(xa))
        if self.fp32_inverse_accumulation and y.dtype != torch.float32:
            xb = (
                yb.float()
                - self._scale("attention").float()
                * attention_update.float()
            ).to(y.dtype)
        else:
            xb = yb - self._scale("attention") * attention_update
        return torch.cat((xa, xb), dim=-1)

    def _mix_output(self, ya, yb):
        if self.output_mixing == "none":
            return torch.cat((ya, yb), dim=-1)
        if self.output_mixing == "swap":
            return torch.cat((yb, ya), dim=-1)
        scale = 2.0**-0.5
        return torch.cat(((ya + yb) * scale, (ya - yb) * scale), dim=-1)

    def _unmix_output(self, y):
        first, second = y.chunk(2, dim=-1)
        if self.output_mixing == "none":
            return first, second
        if self.output_mixing == "swap":
            return second, first
        scale = 2.0**-0.5
        return (first + second) * scale, (first - second) * scale

    def _scale(self, branch):
        logit = getattr(self, f"{branch}_scale_logit", None)
        if logit is None:
            return getattr(self, f"{branch}_scale")
        return self.residual_scale_min + (
            self.residual_scale_max - self.residual_scale_min
        ) * logit.sigmoid()

    def jacobian_probe(self, x, vector=None):
        """Estimate forward and transpose-Jacobian amplification on one sample."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        probe_x = x.detach().requires_grad_(True)
        if vector is None:
            vector = torch.randn_like(probe_x)
        vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
        # CPU flash SDPA does not currently expose the higher-order derivative
        # required by torch.autograd.functional.jvp. Probe with the math kernel
        # while leaving the normal training path unchanged.
        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.autograd.functional.jvp(
                self._forward_impl, probe_x, vector, create_graph=False
            )
            output = self._forward_impl(probe_x)
            (jtv,) = torch.autograd.grad(
                output, probe_x, vector, retain_graph=False
            )
        denominator = vector.float().norm().item()
        return {
            "jv_ratio": jv.float().norm().item() / denominator,
            "jtv_ratio": jtv.float().norm().item() / denominator,
        }

    def branch_jacobian_probe(self, x, branch, vector=None):
        """Estimate the unscaled Attention or FFN branch Jacobian gain."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if branch == "attention":
            function = lambda value: self.attention(
                self.attention_norm(value)
            )
        elif branch == "ffn":
            function = lambda value: self.ffn(self.ffn_norm(value))
        else:
            raise ValueError("branch must be 'attention' or 'ffn'")
        probe_x = x.detach().requires_grad_(True)
        if vector is None:
            vector = torch.randn_like(probe_x)
        vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.autograd.functional.jvp(
                function, probe_x, vector, create_graph=False
            )
            output = function(probe_x)
            (jtv,) = torch.autograd.grad(
                output, probe_x, vector, retain_graph=False
            )
        denominator = vector.float().norm().item()
        return {
            "jv_ratio": jv.float().norm().item() / denominator,
            "jtv_ratio": jtv.float().norm().item() / denominator,
        }

    def spectral_probe(self, x, branch=None, iterations=3, vector=None):
        """Power-iterate J^T J to estimate a block or branch sigma_max."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if iterations < 1:
            raise ValueError("iterations must be positive")
        if branch is None:
            function = self._forward_impl
        elif branch == "attention":
            function = lambda value: self.attention(
                self.attention_norm(value)
            )
        elif branch == "ffn":
            function = lambda value: self.ffn(self.ffn_norm(value))
        else:
            raise ValueError("branch must be None, 'attention', or 'ffn'")
        probe_x = x.detach()
        if vector is None:
            vector = torch.randn_like(probe_x)
        vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
        sigma = 0.0
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(iterations):
                current = probe_x.detach().requires_grad_(True)
                _, jv = torch.autograd.functional.jvp(
                    function, current, vector, create_graph=False
                )
                output = function(current)
                (jtv,) = torch.autograd.grad(
                    output, current, jv.detach(), retain_graph=False
                )
                sigma = float(jv.float().norm())
                vector = jtv.detach()
                vector = vector / vector.float().norm().clamp_min(1e-12).to(
                    vector.dtype
                )
        current = probe_x.detach().requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.autograd.functional.jvp(
                function, current, vector, create_graph=False
            )
        sigma = float(jv.float().norm())
        return {"sigma_max": sigma, "iterations": iterations}


class _ReversibleBlockFunction(torch.autograd.Function):
    """Activation-reconstructing backward for one deterministic block.

    Parameters are explicit Function inputs so autograd receives one gradient
    for every module parameter. Only the block output is retained from forward;
    the input and internal activations are reconstructed during backward.
    """

    @staticmethod
    def forward(ctx, x, block, *parameters):
        ctx.block = block
        ctx.parameters = parameters
        with torch.no_grad():
            y = block._forward_impl(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        block = ctx.block
        parameters = ctx.parameters

        with torch.no_grad():
            reconstructed = block.inverse(y)

        with torch.enable_grad():
            x = reconstructed.detach().requires_grad_(True)
            recomputed = block._forward_impl(x)
            gradients = torch.autograd.grad(
                outputs=recomputed,
                inputs=(x, *parameters),
                grad_outputs=grad_y,
                allow_unused=False,
            )

        return (gradients[0], None, *gradients[1:])


def reversible_sequence(x, blocks):
    """Run consecutive reversible blocks while saving one segment boundary."""
    blocks = tuple(blocks)
    if not blocks:
        return x
    parameters = tuple(
        parameter for block in blocks for parameter in block.parameters()
    )
    return _ReversibleSequenceFunction.apply(x, blocks, *parameters)


def reversible_block_with_int8_correction(x, block):
    """Reconstruct one block while retaining only its quantized inverse error."""
    parameters = tuple(block.parameters())
    return _CorrectedReversibleBlockFunction.apply(x, block, *parameters)


class _CorrectedReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block, *parameters):
        ctx.block = block
        ctx.parameters = parameters
        with torch.no_grad():
            y = block._forward_impl(x)
            reconstructed = block.inverse(y)
            residual = x.float() - reconstructed.float()
            scale = residual.abs().amax().clamp_min(1e-12) / 127.0
            quantized = (residual / scale).round().clamp(-127, 127).to(torch.int8)
        ctx.save_for_backward(y, quantized, scale)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, quantized, scale = ctx.saved_tensors
        block = ctx.block
        parameters = ctx.parameters
        with torch.no_grad():
            reconstructed = block.inverse(y).float()
            reconstructed.add_(quantized.float(), alpha=float(scale))
            reconstructed = reconstructed.to(y.dtype)
        with torch.enable_grad():
            x = reconstructed.detach().requires_grad_(True)
            recomputed = block._forward_impl(x)
            gradients = torch.autograd.grad(
                outputs=recomputed,
                inputs=(x, *parameters),
                grad_outputs=grad_y,
                allow_unused=False,
            )
        return (gradients[0], None, *gradients[1:])


class _ReversibleSequenceFunction(torch.autograd.Function):
    """Activation reconstruction across a consecutive block segment.

    In contrast to ``_ReversibleBlockFunction``, this function retains only the
    final output of the whole segment. Backward walks through every inverse to
    recover the segment input, then recomputes the segment once with autograd.
    """

    @staticmethod
    def forward(ctx, x, blocks, *parameters):
        ctx.blocks = blocks
        ctx.parameters = parameters
        with torch.no_grad():
            y = x
            for block in blocks:
                y = block._forward_impl(y)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        blocks = ctx.blocks
        parameters = ctx.parameters
        parameter_counts = [sum(1 for _ in block.parameters()) for block in blocks]
        offsets = []
        offset = 0
        for count in parameter_counts:
            offsets.append(offset)
            offset += count
        parameter_gradients = [None] * len(parameters)
        current_y = y
        current_grad = grad_y

        # Reconstruct and differentiate one block at a time. This preserves the
        # single saved segment boundary without keeping every recomputed block's
        # internal activation graph alive at once.
        for block, start, count in reversed(
            list(zip(blocks, offsets, parameter_counts))
        ):
            block_parameters = parameters[start : start + count]
            with torch.no_grad():
                reconstructed = block.inverse(current_y)
            with torch.enable_grad():
                x = reconstructed.detach().requires_grad_(True)
                recomputed = block._forward_impl(x)
                gradients = torch.autograd.grad(
                    outputs=recomputed,
                    inputs=(x, *block_parameters),
                    grad_outputs=current_grad,
                    allow_unused=False,
                )
            current_y = reconstructed
            current_grad = gradients[0]
            parameter_gradients[start : start + count] = gradients[1:]

        return (current_grad, None, *parameter_gradients)
