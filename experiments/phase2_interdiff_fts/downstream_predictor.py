"""
Next-day return predictor for CSI800 ranking / IC evaluation.

Takes a panel window (B, N, L, C) of past (log_ret, log_hc, log_lc, log_oc),
outputs (B, N) predicted next-day log_ret scores. Trained by MSE; evaluated
by cross-sectional Spearman rank-IC.

Architecture is intentionally small — a 2-block hierarchical transformer
(same InterBlock as the generator) with a last-timestep readout. We keep
it small so the α-sweep runs in reasonable time on GPU.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model import InterBlock


class NextDayPredictor(nn.Module):
    """
    Input:  x (B, N, L, C)   panel window of past returns
    Output: (B, N)            predicted next-day log_ret (first channel)

    The model only looks at its input window (no leakage). Apply to a
    window ending at day t to predict day t+1.
    """

    def __init__(
        self,
        n_channels: int = 4,
        max_length: int = 32,
        max_stocks: int = 32,
        d_model: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(n_channels, d_model)
        self.t_pos = nn.Parameter(torch.zeros(1, 1, max_length, d_model))
        self.s_pos = nn.Parameter(torch.zeros(1, max_stocks, 1, d_model))
        nn.init.trunc_normal_(self.t_pos, std=0.02)
        nn.init.trunc_normal_(self.s_pos, std=0.02)

        self.blocks = nn.ModuleList(
            [InterBlock(d_model, n_heads, ff_mult=4, dropout=dropout) for _ in range(n_blocks)]
        )
        self.ln_out = nn.LayerNorm(d_model)
        # Readout on last timestep only -> per-stock scalar
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, C = x.shape
        h = self.in_proj(x)
        h = h + self.t_pos[:, :, :L, :] + self.s_pos[:, :N, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_out(h)
        # Use last time step as summary of the history
        last = h[:, :, -1, :]  # (B, N, d)
        return self.head(last).squeeze(-1)  # (B, N)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def rank_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-sectional Spearman rank correlation, averaged over batch.

    pred, target:  (B, N)
    returns:       scalar mean IC across the B panels
    """
    # For each row, argsort -> ranks (float)
    def _ranks(z: torch.Tensor) -> torch.Tensor:
        # z: (B, N)
        B, N = z.shape
        order = z.argsort(dim=-1)
        ranks = torch.zeros_like(z, dtype=torch.float32)
        r = torch.arange(N, dtype=torch.float32, device=z.device).expand(B, -1)
        ranks.scatter_(-1, order, r)
        return ranks

    rp = _ranks(pred)
    rt = _ranks(target)
    rp = rp - rp.mean(dim=-1, keepdim=True)
    rt = rt - rt.mean(dim=-1, keepdim=True)
    num = (rp * rt).sum(dim=-1)
    den = torch.sqrt((rp * rp).sum(dim=-1) * (rt * rt).sum(dim=-1) + 1e-12)
    return (num / den).mean()


def ic_from_numpy(pred_np, target_np) -> float:
    """
    Simple Spearman IC for 1-D arrays (one day): corr(rank(pred), rank(actual)).
    Returns NaN if not enough valid values.
    """
    import numpy as np
    mask = np.isfinite(pred_np) & np.isfinite(target_np)
    if mask.sum() < 5:
        return float("nan")
    p = pred_np[mask]
    a = target_np[mask]
    pr = p.argsort().argsort().astype(float)
    ar = a.argsort().argsort().astype(float)
    pr -= pr.mean(); ar -= ar.mean()
    d = (pr.std() * ar.std())
    if d <= 0:
        return float("nan")
    return float((pr * ar).mean() / d)


if __name__ == "__main__":
    m = NextDayPredictor(n_channels=4, max_length=32, max_stocks=32)
    print(f"params: {count_params(m):,}")
    x = torch.randn(4, 32, 32, 4)
    out = m(x)
    print(f"out: {out.shape}")
    # IC sanity
    y = torch.randn(4, 32)
    print(f"IC(pred, y): {rank_ic(out, y).item():.4f}")
    print(f"IC(y, y):    {rank_ic(y, y).item():.4f}  (should be 1.0)")
