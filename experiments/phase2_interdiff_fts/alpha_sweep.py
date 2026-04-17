"""
Alpha-sweep: Train a next-day return predictor with (1-alpha) real +
alpha synthetic data mixing; evaluate cross-sectional rank-IC on
held-out 2024 real test set.

Sweeps alpha ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9} over multiple seeds and
records IC / IC_IR per setting. If M6 synthetic data is useful for
downstream prediction, small alpha should improve IC over alpha=0;
if the generator suffers model collapse when alpha grows, IC should
degrade past some threshold alpha* (the phase-transition point).

Usage:
    python alpha_sweep.py \
        --panel data/csi800_2015_2024.npz \
        --synth ckpts/M0_m6_csi800_step20000.samples.npz \
        --alphas 0,0.1,0.25,0.5,0.75,0.9 \
        --seeds 0,1,2 \
        --steps 3000
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from downstream_predictor import NextDayPredictor, count_params, ic_from_numpy
from panel_windows import derive_panel_returns


# ────────────────────────────────────────────────────────────────
# Real data preparation (no normalization — use raw log_ret space)
# ────────────────────────────────────────────────────────────────

def load_real_returns(panel_npz: str) -> dict:
    """
    Load real CSI800 panel and return raw log_ret + auxiliary channels.

    Returns:
        returns:   (N_stocks, T, 4) raw log returns [log_ret, log_hc, log_lc, log_oc]
        valid:     (N_stocks, T) bool mask
        dates:     (T,) str
        codes:     (N_stocks,) str
    """
    d = np.load(panel_npz, allow_pickle=True)
    panel = d["panel"]
    fields = d["fields"].tolist()
    dates = d["dates"].astype(str)

    returns = derive_panel_returns(panel, fields)
    ret_dates = dates[1:]
    valid = np.isfinite(returns).all(axis=2)
    returns_clean = np.where(valid[:, :, None], returns, 0.0).astype(np.float32)

    return {
        "returns": returns_clean,  # (N, T, 4) raw log_ret
        "valid": valid,            # (N, T)
        "dates": ret_dates,        # (T,)
        "codes": d["codes"],
    }


def compute_valid_starts(valid: np.ndarray, length: int) -> np.ndarray:
    """
    For each start day s, compute mask (N,) of stocks with fully-valid
    window [s, s+length). Returns (T-length+1, N) boolean array.
    """
    N, T = valid.shape
    cum = np.zeros((N, T + 1), dtype=np.int32)
    cum[:, 1:] = np.cumsum(valid.astype(np.int32), axis=1)
    # full[s, i] = all of [s, s+length) valid for stock i
    full = (cum[:, length:] - cum[:, :T - length + 1]) == length  # (N, T-length+1)
    return full.T  # (T-length+1, N) for convenience


# ────────────────────────────────────────────────────────────────
# Batch samplers
# ────────────────────────────────────────────────────────────────

def sample_real_batch(
    returns: np.ndarray,       # (N, T, C)
    full_mask: np.ndarray,     # (n_valid_starts, N) bool
    valid_starts: np.ndarray,  # (n_valid_starts,) day indices
    B: int, N_pick: int, L: int, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample B panels of (N_pick stocks × L history + 1-day label) from real."""
    N_total, T, C = returns.shape
    out_x = np.empty((B, N_pick, L, C), dtype=np.float32)
    out_y = np.empty((B, N_pick), dtype=np.float32)
    for b in range(B):
        # Pick a valid start (window length L+1)
        # full_mask is for length L+1 windows
        while True:
            s_idx = int(rng.integers(0, len(valid_starts)))
            s = int(valid_starts[s_idx])
            ok = np.where(full_mask[s_idx])[0]
            if ok.size >= N_pick:
                break
        picks = rng.choice(ok, N_pick, replace=False)
        out_x[b] = returns[picks, s:s + L, :]
        out_y[b] = returns[picks, s + L, 0]  # next-day log_ret
    return torch.from_numpy(out_x), torch.from_numpy(out_y)


def sample_synth_batch(
    synth_panels: np.ndarray,  # (n_panels, K, T_syn, C)  raw log_ret
    B: int, N_pick: int, L: int, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample B panels of (N_pick × L + 1-day label) from synthetic corpus."""
    n_panels, K, T_syn, C = synth_panels.shape
    out_x = np.empty((B, N_pick, L, C), dtype=np.float32)
    out_y = np.empty((B, N_pick), dtype=np.float32)
    for b in range(B):
        p = int(rng.integers(0, n_panels))
        start = int(rng.integers(0, T_syn - L - 1))
        if N_pick <= K:
            stock_idx = rng.choice(K, N_pick, replace=False)
        else:
            raise ValueError(f"N_pick ({N_pick}) > synth K ({K})")
        out_x[b] = synth_panels[p, stock_idx, start:start + L, :]
        out_y[b] = synth_panels[p, stock_idx, start + L, 0]
    return torch.from_numpy(out_x), torch.from_numpy(out_y)


# ────────────────────────────────────────────────────────────────
# Test IC evaluation
# ────────────────────────────────────────────────────────────────

def eval_test_ic(
    model: torch.nn.Module,
    test_returns: np.ndarray,  # (N, T_test, C)
    test_valid: np.ndarray,    # (N, T_test)
    N_pick: int, L: int, device: str,
) -> dict:
    """
    For each day t in [L-1, T_test-1), build a panel of top-N_pick valid
    stocks from days t-L+1..t, predict day t+1, compute rank-IC vs actual.
    Returns dict with mean, std, IR = mean/std.
    """
    model.eval()
    N_total, T, C = test_returns.shape
    ics = []
    skipped = 0
    # We need 33-day windows (L history + 1 label). Valid = all days in window valid.
    with torch.no_grad():
        for t in range(L, T - 1):
            # Window: days [t-L, t) history, day t label
            # Check which stocks have full window valid
            window_valid = test_valid[:, t - L:t].all(axis=1) & test_valid[:, t]
            ok = np.where(window_valid)[0]
            if ok.size < N_pick:
                skipped += 1
                continue
            picks = ok[:N_pick]  # take first N_pick (deterministic)
            x = test_returns[picks, t - L:t, :]  # (N_pick, L, C)
            y_true = test_returns[picks, t, 0]   # (N_pick,)

            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, N, L, C)
            pred = model(x_t).squeeze(0).cpu().numpy()          # (N_pick,)

            ic = ic_from_numpy(pred, y_true)
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    return {
        "ic_mean": float(ics.mean()) if ics.size else float("nan"),
        "ic_std": float(ics.std()) if ics.size else float("nan"),
        "ic_ir": float(ics.mean() / (ics.std() + 1e-9)) if ics.size else float("nan"),
        "n_days": int(ics.size),
        "n_skipped": skipped,
    }


# ────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────

def run_one(
    alpha: float, seed: int, steps: int,
    real_train: dict, real_test: dict, synth_panels: np.ndarray,
    train_full_mask: np.ndarray, train_valid_starts: np.ndarray,
    N_pick: int, L: int, batch_size: int,
    d_model: int, n_blocks: int, n_heads: int,
    lr: float, device: str,
) -> dict:
    """Train one predictor and evaluate its test IC. Returns summary dict."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = NextDayPredictor(
        n_channels=4, max_length=L, max_stocks=N_pick,
        d_model=d_model, n_blocks=n_blocks, n_heads=n_heads, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        use_synth = (alpha > 0) and (rng.random() < alpha)
        if use_synth:
            x, y = sample_synth_batch(synth_panels, batch_size, N_pick, L, rng)
        else:
            x, y = sample_real_batch(
                real_train["returns"], train_full_mask, train_valid_starts,
                batch_size, N_pick, L, rng,
            )
        x = x.to(device); y = y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))

    train_time = time.time() - t0
    ic = eval_test_ic(model, real_test["returns"], real_test["valid"],
                     N_pick, L, device)
    return {
        "alpha": alpha,
        "seed": seed,
        "steps": steps,
        "train_time_s": round(train_time, 1),
        "final_train_loss": round(float(np.mean(losses[-100:])), 6),
        **ic,
    }


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/csi800_2015_2024.npz")
    ap.add_argument("--synth", default="ckpts/M0_m6_csi800_step20000.samples.npz")
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--N-pick", type=int, default=32)
    ap.add_argument("--L", type=int, default=32, help="history length")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--out", default="alpha_sweep_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sweep] device={device}  alphas={alphas}  seeds={seeds}  steps={args.steps}")

    # --- load real data ---
    print(f"[sweep] loading real panel {args.panel}")
    real = load_real_returns(args.panel)
    print(f"[sweep]   shape (N, T, C) = {real['returns'].shape}  "
          f"date range {real['dates'][0]} -> {real['dates'][-1]}")

    # Train/test split
    train_mask = real["dates"] <= args.train_end
    test_mask = (real["dates"] >= args.test_start) & (real["dates"] <= args.test_end)
    real_train = {
        "returns": real["returns"][:, train_mask, :],
        "valid": real["valid"][:, train_mask],
        "dates": real["dates"][train_mask],
    }
    real_test = {
        "returns": real["returns"][:, test_mask, :],
        "valid": real["valid"][:, test_mask],
        "dates": real["dates"][test_mask],
    }
    print(f"[sweep]   train: {real_train['returns'].shape[1]} days  "
          f"test: {real_test['returns'].shape[1]} days")

    # Precompute train window masks (for L+1 length)
    window_len = args.L + 1
    train_full = compute_valid_starts(real_train["valid"], window_len)  # (n_starts, N)
    # Only keep starts with at least N_pick valid stocks
    enough = train_full.sum(axis=1) >= args.N_pick
    train_valid_starts = np.where(enough)[0]
    train_full_reduced = train_full[enough]
    print(f"[sweep]   train valid starts: {len(train_valid_starts)}  "
          f"mean stocks/start: {train_full_reduced.sum(1).mean():.0f}")

    # --- load synth ---
    print(f"[sweep] loading synth {args.synth}")
    d_syn = np.load(args.synth, allow_pickle=True)
    synth_panels = d_syn["panels_denorm"].astype(np.float32)
    print(f"[sweep]   synth panels shape: {synth_panels.shape}")

    # --- run sweep ---
    results = []
    for alpha in alphas:
        for seed in seeds:
            print(f"\n[sweep] === alpha={alpha}  seed={seed} ===")
            r = run_one(
                alpha=alpha, seed=seed, steps=args.steps,
                real_train=real_train, real_test=real_test,
                synth_panels=synth_panels,
                train_full_mask=train_full_reduced,
                train_valid_starts=train_valid_starts,
                N_pick=args.N_pick, L=args.L, batch_size=args.batch,
                d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads,
                lr=args.lr, device=device,
            )
            results.append(r)
            print(f"[sweep]   train_loss={r['final_train_loss']:.5f}  "
                  f"IC={r['ic_mean']:+.5f} +- {r['ic_std']:.5f}  "
                  f"IC_IR={r['ic_ir']:+.3f}  "
                  f"n_days={r['n_days']}  time={r['train_time_s']}s")

    # Save and summarise
    out = {"config": vars(args), "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[sweep] saved -> {args.out}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'alpha':>6s}  {'IC mean':>9s} ± {'std':>7s}  {'IC_IR':>7s}  {'n_seeds':>7s}")
    print("-" * 70)
    for a in alphas:
        rs = [r for r in results if r["alpha"] == a]
        ic_means = [r["ic_mean"] for r in rs if np.isfinite(r["ic_mean"])]
        ic_irs = [r["ic_ir"] for r in rs if np.isfinite(r["ic_ir"])]
        if ic_means:
            print(f"{a:>6.2f}  {np.mean(ic_means):+9.5f} ± {np.std(ic_means):7.5f}  "
                  f"{np.mean(ic_irs):+7.3f}  {len(ic_means):>7d}")


if __name__ == "__main__":
    main()
