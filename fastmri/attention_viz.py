import numpy as np, torch, torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Subset
import fastmri
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform


@torch.no_grad()
def attention_figures(sn, k, m, nlf, ncoils=6):
    """sn: SensitivityModel with coil_attn. Returns (overlay_fig, matrix_fig, stats)."""
    if torch.is_tensor(nlf):                 # <- new
        nlf = nlf.to(k.device)               # <- new
    ca = sn.coil_attn
    try:
        ca.store = True
        S = sn(k, m, nlf); A = ca.last_attn                 # maps with attention; weights (h*w, C, C)
        ca.store = False
        sn.coil_attn = None
        S0 = sn(k, m, nlf)                                  # same weights, layer switched off
    finally:
        ca.store = False
        sn.coil_attn = ca
    zf = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(k)), dim=1)[0]
    C, H, W = S.shape[1:4]; h, w = H // ca.pool, W // ca.pool
    A = A.reshape(h, w, C, C)                               # [y, x, query coil, key coil]
    recv = A.mean(dim=2).permute(2, 0, 1)                   # attention each coil receives
    recv = F.interpolate(recv[None], size=(H, W), mode="bilinear", align_corners=False)[0] * C  # 1 = uniform
    mag, mag0 = fastmri.complex_abs(S)[0], fastmri.complex_abs(S0)[0]
    fg = zf > 0.1 * zf.max()
    fgp = F.adaptive_avg_pool2d(fg[None, None].float(), (h, w))[0, 0] > 0.5
    Afg = A[fgp] if fgp.any() else A.reshape(-1, C, C)
    mat = Afg.mean(0)
    ent = float((-(Afg * (Afg + 1e-9).log()).sum(-1)).mean() / np.log(C))
    rel = float((mag - mag0).abs().mean() / mag.mean())

    cs = np.linspace(0, C - 1, ncoils).astype(int); fgn = fg.cpu().numpy()
    fig = Figure(figsize=(2.6 * len(cs), 8.2)); ax = fig.subplots(3, len(cs), squeeze=False)
    for j, c in enumerate(cs):
        mc = mag[c].cpu().numpy()
        ax[0, j].imshow(mc, cmap="gray"); ax[0, j].set_title(f"coil {c}")
        ax[1, j].imshow(mc, cmap="gray")
        im1 = ax[1, j].imshow(np.ma.masked_where(~fgn, recv[c].cpu().numpy()),
                              cmap="coolwarm", vmin=0, vmax=2, alpha=0.55)
        d = (mag[c] - mag0[c]).cpu().numpy(); lim = max(np.abs(d).max(), 1e-8)
        im2 = ax[2, j].imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
        for r in range(3): ax[r, j].axis("off")
    fig.text(0.005, 0.80, "|S| (with attention)", rotation=90, va="center")
    fig.text(0.005, 0.50, "attention received", rotation=90, va="center")
    fig.text(0.005, 0.20, "effect of layer", rotation=90, va="center")
    fig.colorbar(im1, ax=ax[1, :].tolist(), shrink=0.8, label="attention received (1 = uniform)")
    fig.colorbar(im2, ax=ax[2, :].tolist(), shrink=0.8, label="|S| with minus without attention")

    fig2 = Figure(figsize=(5, 4.5)); ax2 = fig2.subplots()
    im = ax2.imshow(mat.cpu().numpy(), cmap="viridis")
    ax2.set_xlabel("key coil (attended to)"); ax2.set_ylabel("query coil")
    ax2.set_title(f"mean coil-to-coil attention (uniform = {1/C:.3f})"); fig2.colorbar(im)
    return fig, fig2, {"entropy": ent, "effect": rel}


class CoilAttentionLogger(pl.Callback):
    def __init__(self, data_dir, mask_type="random", center_fractions=(0.08,),
                 accelerations=(4,), slice_num=8, ncoils=6, every_n_epochs=2):
        self.data_dir, self.mask_type = data_dir, mask_type
        self.cf, self.acc = list(center_fractions), list(accelerations)
        self.slice_num, self.ncoils, self.every = slice_num, ncoils, every_n_epochs
        self.batch = None

    def _load(self):
        mask = create_mask_for_mask_type(self.mask_type, self.cf, self.acc)
        ds = SliceDataset(root=self.data_dir, challenge="multicoil",
                          transform=VarNetDataTransform(mask_func=mask, use_seed=True))
        for i in range(len(ds)):
            if ds[i].slice_num == self.slice_num:           # first volume that has this slice
                return next(iter(DataLoader(Subset(ds, [i]), batch_size=1)))
        raise RuntimeError("slice not found for attention logging")

    def on_validation_epoch_end(self, trainer, pl_module):
        sn = pl_module.varnet.sens_net
        if sn.coil_attn is None or trainer.sanity_checking or trainer.current_epoch % self.every:
            return
        writer = getattr(trainer.logger, "experiment", None)
        if writer is None or not hasattr(writer, "add_figure"):
            return
        if self.batch is None:
            self.batch = self._load()
        b, dev = self.batch, pl_module.device
        was = pl_module.training; pl_module.eval()
        try:
            f1, f2, st = attention_figures(sn, b.masked_kspace.to(dev), b.mask.to(dev),
                                           b.num_low_frequencies, self.ncoils)
        finally:
            pl_module.train(was)
        step = trainer.global_step
        writer.add_figure("attention/overlay", f1, step)
        writer.add_figure("attention/coil_matrix", f2, step)
        writer.add_scalar("attention/entropy_norm", st["entropy"], step)       # 1 = uniform attention
        writer.add_scalar("attention/layer_effect_rel", st["effect"], step)    # how much the layer changes |S|
        writer.add_scalar("attention/out_weight_norm", float(sn.coil_attn.out.weight.norm()), step)