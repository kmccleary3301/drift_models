from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiTLikeConfig:
    image_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    hidden_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    ffn_inner_dim: int | None = None
    num_classes: int = 1000
    register_tokens: int = 16
    style_vocab_size: int = 64
    style_token_count: int = 32
    alpha_hidden_dim: int = 128
    norm_type: str = "layernorm"
    use_qk_norm: bool = False
    use_rope: bool = False


class DiTLikeGenerator(nn.Module):
    def __init__(self, config: DiTLikeConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.patch_positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, config.hidden_dim) * 0.02
        )

        self.class_embedding = nn.Embedding(config.num_classes, config.hidden_dim)
        self.alpha_embedding = nn.Sequential(
            nn.Linear(1, config.alpha_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.alpha_hidden_dim, config.hidden_dim),
        )
        self.style_embedding = nn.Embedding(config.style_vocab_size, config.hidden_dim)

        self.register_tokens = config.register_tokens
        if self.register_tokens > 0:
            self.register_base = nn.Parameter(torch.randn(1, self.register_tokens, config.hidden_dim) * 0.02)
            self.register_positional_embedding = nn.Parameter(
                torch.randn(1, self.register_tokens, config.hidden_dim) * 0.02
            )
            self.condition_to_register = nn.Linear(config.hidden_dim, config.hidden_dim)
        else:
            self.register_base = None
            self.register_positional_embedding = None
            self.condition_to_register = None

        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                ffn_inner_dim=config.ffn_inner_dim,
                norm_type=config.norm_type,
                use_qk_norm=config.use_qk_norm,
                use_rope=config.use_rope,
            )
            for _ in range(config.depth)
        )
        self.final_norm = make_norm(config.hidden_dim, config.norm_type)
        self.output_projection = nn.Linear(
            config.hidden_dim,
            config.patch_size * config.patch_size * config.out_channels,
        )

    def forward(
        self,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        alpha: torch.Tensor,
        style_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise.ndim != 4:
            raise ValueError(f"noise must be [B, C, H, W], got {tuple(noise.shape)}")
        if class_labels.ndim != 1 or class_labels.shape[0] != noise.shape[0]:
            raise ValueError("class_labels must be [B]")
        if alpha.ndim != 1 or alpha.shape[0] != noise.shape[0]:
            raise ValueError("alpha must be [B]")

        batch = noise.shape[0]
        patch_tokens = self.patch_embed(noise)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        patch_tokens = patch_tokens + self.patch_positional_embedding

        condition = self._build_conditioning(
            class_labels=class_labels,
            alpha=alpha,
            style_indices=style_indices,
            device=noise.device,
            batch=batch,
        )

        if self.register_tokens > 0:
            register_tokens = self.register_base + self.register_positional_embedding
            register_tokens = register_tokens.repeat(batch, 1, 1)
            register_tokens = register_tokens + self.condition_to_register(condition).unsqueeze(1)
            tokens = torch.cat([register_tokens, patch_tokens], dim=1)
        else:
            tokens = patch_tokens

        for block in self.blocks:
            tokens = block(tokens, condition)

        if self.register_tokens > 0:
            tokens = tokens[:, self.register_tokens :, :]
        tokens = self.final_norm(tokens)
        patch_values = self.output_projection(tokens)
        return self._unpatchify(patch_values)

    def _build_conditioning(
        self,
        *,
        class_labels: torch.Tensor,
        alpha: torch.Tensor,
        style_indices: torch.Tensor | None,
        device: torch.device,
        batch: int,
    ) -> torch.Tensor:
        class_cond = self.class_embedding(class_labels)
        alpha_cond = self.alpha_embedding(alpha.unsqueeze(-1))
        if style_indices is None:
            style_indices = torch.zeros(
                batch,
                self.config.style_token_count,
                device=device,
                dtype=torch.long,
            )
        if style_indices.ndim != 2:
            raise ValueError("style_indices must be [B, S]")
        style_cond = self.style_embedding(style_indices).sum(dim=1)
        return class_cond + alpha_cond + style_cond

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        batch = patch_values.shape[0]
        patch = self.config.patch_size
        out_channels = self.config.out_channels
        patch_values = patch_values.view(batch, self.grid_size, self.grid_size, patch, patch, out_channels)
        patch_values = patch_values.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patch_values.view(batch, out_channels, self.config.image_size, self.config.image_size)


class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        ffn_inner_dim: int | None,
        norm_type: str,
        use_qk_norm: bool,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.norm1 = make_norm(hidden_dim, norm_type)
        self.norm2 = make_norm(hidden_dim, norm_type)
        self.attention = DiTAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
        )
        self.feedforward = SwiGLUFeedForward(hidden_dim=hidden_dim, mlp_ratio=mlp_ratio, inner_dim=ffn_inner_dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        modulation = self.modulation(condition)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        norm_attn = _modulate(self.norm1(tokens), shift_attn, scale_attn)
        attn_out = self.attention(norm_attn)
        tokens = tokens + gate_attn.unsqueeze(1) * attn_out

        norm_mlp = _modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        mlp_out = self.feedforward(norm_mlp)
        tokens = tokens + gate_mlp.unsqueeze(1) * mlp_out
        return tokens


class SwiGLUFeedForward(nn.Module):
    def __init__(self, *, hidden_dim: int, mlp_ratio: float, inner_dim: int | None = None) -> None:
        super().__init__()
        if inner_dim is None:
            # For SwiGLU, the input projection is twice the inner size (value+gate),
            # so the parameter count is ~1.5x a standard 2-layer MLP at the same
            # `inner_dim`. The paper describes a DiT-style transformer with SwiGLU,
            # but does not specify the intermediate dimension explicitly. We default
            # to scaling by 2/3 so the FFN parameter budget matches a standard MLP
            # with the same `mlp_ratio`.
            inner_dim = int(hidden_dim * mlp_ratio * (2.0 / 3.0))
        self.linear_in = nn.Linear(hidden_dim, inner_dim * 2)
        self.linear_out = nn.Linear(inner_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value, gate = self.linear_in(tokens).chunk(2, dim=-1)
        return self.linear_out(value * torch.nn.functional.silu(gate))


def _modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTAttention(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, use_qk_norm: bool, use_rope: bool) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if use_rope and (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("head_dim must be even when using RoPE")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.view(batch, length, 3, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        q, k, v = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]
        if self.use_rope:
            q, k = apply_rope(q, k)
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.hidden_dim)
        return self.out(out)


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(torch.mean(tokens * tokens, dim=-1, keepdim=True) + self.eps)
        return tokens * norm * self.weight


def make_norm(hidden_dim: int, norm_type: str) -> nn.Module:
    normalized = norm_type.lower()
    if normalized == "layernorm":
        return nn.LayerNorm(hidden_dim, elementwise_affine=False)
    if normalized == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = q.shape[-2]
    head_dim = q.shape[-1]
    half = head_dim // 2
    device = q.device
    dtype = q.dtype
    positions = torch.arange(length, device=device, dtype=dtype)
    frequencies = torch.arange(half, device=device, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (frequencies / max(1, half)))
    theta = torch.outer(positions, inv_freq)
    sin = torch.sin(theta).view(1, 1, length, half)
    cos = torch.cos(theta).view(1, 1, length, half)
    return _rotate(q, sin, cos), _rotate(k, sin, cos)


def _rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
