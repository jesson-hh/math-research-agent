"""
Cross-section gap diagnosis for M1 vs real (L=64 window space).

Decomposes panel corr into:
  1) market factor fraction (top eigenvalue / k)
  2) residual structure (mean pair corr after market-factor regression)
  3) per-time cross-sectional std (how much do stocks move together vs independently)
"""
import numpy as np
import torch

from panel_windows import PanelWindowDataset
from regimes import RegimeSpec

import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "m1_big"
CK = f"ckpts/M0_{TAG}_step20000.pt"
SAMPLES = f"ckpts/M0_{TAG}_step20000.samples.npz"
print(f"=== {TAG} ===")

ck = torch.load(CK, map_location="cpu", weights_only=False)
spec = RegimeSpec.from_dict(ck["regime_spec"])
cfg = ck["args"]
L = cfg["length"]
K = cfg["k"]

d = np.load(SAMPLES, allow_pickle=True)
syn = d["panels"][..., 0]  # (n_panels, k, L)  NORMALISED space
print(f"syn (normalised) {syn.shape}")

ds = PanelWindowDataset(
    panel_npz="data/csi300_2015_2024.npz",
    length=L, k_stocks=K, seed=0, normalise=True,
    time_range=("2015-01-05", cfg.get("train_end", "2022-12-31")),
    regime_spec=spec,
)
it = iter(ds)
n_p = syn.shape[0]
real = np.zeros((n_p, K, L), dtype=np.float32)
for i in range(n_p):
    item = next(it)
    w = item[0] if isinstance(item, (list, tuple)) else item
    real[i] = w.numpy()[..., 0]


def _decompose(panels):
    """panels: (P, k, L) — treat each panel as independent."""
    eigs = []
    mean_corrs = []
    resid_mean_corrs = []
    market_vars = []
    per_t_std = []
    for p in panels:
        if p.std() == 0:
            continue
        C = np.corrcoef(p)
        if not np.isfinite(C).all():
            continue
        iu = np.triu_indices_from(C, k=1)
        mean_corrs.append(float(np.mean(C[iu])))
        w = np.linalg.eigvalsh(C)
        eigs.append(float(w[-1]))

        mkt = p.mean(axis=0)
        mkt_c = mkt - mkt.mean()
        pc = p - p.mean(axis=1, keepdims=True)
        denom = (mkt_c ** 2).sum() + 1e-12
        beta = (pc @ mkt_c) / denom
        resid = pc - beta[:, None] * mkt_c[None, :]
        Cr = np.corrcoef(resid)
        if np.isfinite(Cr).all():
            resid_mean_corrs.append(float(np.mean(Cr[iu])))
        market_vars.append(float(mkt.var()))
        per_t_std.append(float(p.std(axis=0).mean()))

    return {
        "max_eig_mean": float(np.mean(eigs)),
        "max_eig_frac": float(np.mean(eigs) / panels.shape[1]),
        "mean_pair_corr": float(np.mean(mean_corrs)),
        "resid_mean_pair_corr": float(np.mean(resid_mean_corrs)) if resid_mean_corrs else float("nan"),
        "market_factor_var": float(np.mean(market_vars)),
        "per_t_cross_std": float(np.mean(per_t_std)),
    }


r_stat = _decompose(real)
s_stat = _decompose(syn)

print()
print(f"{'metric':32s}  {'real':>10s}  {'syn':>10s}   gap")
for k_ in ["mean_pair_corr", "max_eig_frac", "resid_mean_pair_corr",
           "market_factor_var", "per_t_cross_std"]:
    r = r_stat[k_]; s = s_stat[k_]
    gap = (s - r) / max(abs(r), 1e-12)
    print(f"{k_:32s}  {r:>10.4f}  {s:>10.4f}   {gap:+.1%}")
