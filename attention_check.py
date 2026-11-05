import torch
from fastmri.models.varnet import VarNet

torch.manual_seed(0)
a = VarNet(num_cascades=2, chans=8, sens_chans=4, coil_attention=False).eval()
b = VarNet(num_cascades=2, chans=8, sens_chans=4, coil_attention=True).eval()
print(b.load_state_dict(a.state_dict(), strict=False))   # only coil_attn keys should be missing

k = torch.randn(1, 8, 64, 64, 2)                        # (batch, coils, H, W, real/imag)
mask = torch.zeros(1, 1, 1, 64, 1, dtype=torch.bool); mask[..., ::4, :] = True; mask[..., 28:36, :] = True
with torch.no_grad():
    print((a(k * mask, mask) - b(k * mask, mask)).abs().max())   # should be ~0