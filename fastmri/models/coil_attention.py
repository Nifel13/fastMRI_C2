import torch
import torch.nn as nn
import torch.nn.functional as F


class CoilAttention(nn.Module):
    """Post-U-Net self-attention across coils at each pooled image pixel."""

    def __init__(self, dim=32, heads=4, pool=8):
        super().__init__()
        self.pool = pool
        self.embed = nn.Conv2d(2, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.out = nn.Conv2d(dim, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        batch, coils, height, width, _ = x.shape
        t = x.permute(0, 1, 4, 2, 3).reshape(
            batch * coils, 2, height, width
        )
        f = self.embed(F.avg_pool2d(t, self.pool))
        d, h, w = f.shape[1:]
        f = (
            f.view(batch, coils, d, h * w)
            .permute(0, 3, 1, 2)
            .reshape(batch * h * w, coils, d)
        )
        y = self.norm1(f)
        f = f + self.attn(y, y, y, need_weights=False)[0]
        f = f + self.mlp(self.norm2(f))
        f = (
            f.view(batch, h * w, coils, d)
            .permute(0, 2, 3, 1)
            .reshape(batch * coils, d, h, w)
        )
        delta = F.interpolate(
            self.out(f),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return x + delta.view(batch, coils, 2, height, width).permute(
            0, 1, 3, 4, 2
        )


class BottleneckCoilAttention(nn.Module):
    """Gated self-attention across coils in U-Net bottleneck feature space.

    Input shape is ``(batch, coils, features, height, width)``. At every
    bottleneck spatial location, coils form the attention sequence. The
    zero-initialized scalar gate makes this module an exact identity when an
    ablation run starts.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                f"Feature dimension {dim} must be divisible by {heads} heads"
            )

        hidden_dim = mlp_ratio * dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, coils, features, height, width = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, coils, features
        )

        normalized = self.norm1(tokens)
        attention_update = self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        contextualized = tokens + attention_update
        mlp_update = self.mlp(self.norm2(contextualized))
        tokens = tokens + torch.tanh(self.gate) * (attention_update + mlp_update)

        return (
            tokens.reshape(batch, height, width, coils, features)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
