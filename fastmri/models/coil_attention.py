import torch, torch.nn as nn, torch.nn.functional as F

class CoilAttention(nn.Module):
    """Self-attention across coils at each (pooled) pixel. x: (B, C, H, W, 2)."""
    def __init__(self, dim=32, heads=4, pool=8):
        super().__init__()
        self.pool = pool
        self.embed = nn.Conv2d(2, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.out = nn.Conv2d(dim, 2, 1)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W, _ = x.shape
        t = x.permute(0, 1, 4, 2, 3).reshape(B * C, 2, H, W)
        f = self.embed(F.avg_pool2d(t, self.pool))              # (B*C, d, h, w)
        d, h, w = f.shape[1:]
        f = f.view(B, C, d, h * w).permute(0, 3, 1, 2).reshape(B * h * w, C, d)
        y = self.norm1(f)
        f = f + self.attn(y, y, y, need_weights=False)[0]       # attention over coils
        f = f + self.mlp(self.norm2(f))
        f = f.view(B, h * w, C, d).permute(0, 2, 3, 1).reshape(B * C, d, h, w)
        delta = F.interpolate(self.out(f), size=(H, W), mode="bilinear", align_corners=False)
        return x + delta.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2)