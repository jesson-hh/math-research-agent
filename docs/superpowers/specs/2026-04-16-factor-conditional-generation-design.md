# Factor-Conditional Panel Generation

> Date: 2026-04-16
> Status: approved
> Direction: improve cross-sectional correlation fidelity in InterDiff synthetic panels

## Problem

Current InterDiff models (m0-m3, 20k steps on CSI300) produce panels where:
- **market_factor_var** is 22% below real (0.272 vs 0.348)
- **panel_mean_pairwise_corr** is 15% below real (0.272-0.282 vs 0.332)
- **max_eig_frac** is 14% below real (0.315-0.328 vs 0.381)

Root cause: the denoiser treats each stock's noise prediction nearly independently. The inter-stock attention layers learn some correlation, but not enough of the shared market movement.

Residual structure after market-factor regression is fine (gap < 4%), so the problem is specifically the common factor.

## Solution: Two-Stage Factor-Conditional Generation

### Stage A: Market Factor Conditioning (do first)

#### Training
1. For each panel window (B, N, L, C), compute the market factor:
   `mkt = x[:, :, :, 0].mean(dim=1)` -> (B, L) — equal-weight cross-sectional mean of log returns
2. Feed `mkt` as an extra conditioning signal into InterDenoiser via FiLM modulation at each InterBlock
3. The denoiser now predicts noise given (noised panel, timestep, regime labels, market factor)

#### Sampling
1. Fit a simple generative model for market factor sequences:
   - Option 1: AR(1) on the real market factor series (fast, interpretable)
   - Option 2: Bootstrap random L-length windows from the real market factor (non-parametric, preserves distribution exactly)
   - Start with bootstrap (simpler), switch to AR if we need novel sequences
2. Sample a market factor sequence mkt ~ p(mkt)
3. Condition InterDiff on mkt to generate the panel

#### Model Changes (model.py)

```python
# Add to InterDenoiser.__init__:
self.mkt_proj = nn.Sequential(
    nn.Linear(1, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)

# Add to InterDenoiser.forward:
# mkt_cond: (B, L) market factor sequence
if mkt_cond is not None:
    mkt_emb = self.mkt_proj(mkt_cond[:, None, :, None].expand(-1, N, -1, 1))
    # mkt_emb: (B, N, L, d_model) — same signal broadcast to all stocks
    h = h + mkt_emb
```

#### Data Pipeline Changes (panel_windows.py)

Each `__iter__` yield becomes `(window, regime_labels, mkt_factor)` where `mkt_factor` is the equal-weight mean of the first channel (log_ret) across the k sampled stocks.

#### Training Changes (train.py)

- Accept mkt_factor from the dataset
- Pass to `model(xt, t, cond=regime_cond, mkt_cond=mkt_factor)`
- Loss unchanged (standard MSE on noise prediction)

#### Sampling Changes (sample.py)

- Bootstrap market factor windows from the real panel
- Pass as mkt_cond during reverse diffusion

#### Verification

Run eval_compare + _diag_cs on the new samples. Success criteria:
- market_factor_var gap < 10% (from current 22%)
- panel_mean_pairwise_corr gap < 8% (from current 15%)
- All existing OK metrics must not degrade to FAIL

### Stage B: Multi-Factor (after A validates)

Only proceed if Stage A narrows the gap significantly.

#### Industry Factor Extraction
1. Load industry labels from `G:/stocks/stock_data/parquet/tushare_stock_basic.parquet`
2. Map tushare `ts_code` (e.g. `000001.SZ`) to qlib code format (e.g. `sz000001`)
3. Collapse 110 fine-grained industries into ~10 sector groups (e.g. finance, tech, consumer, industrial, materials, healthcare, energy, utilities, telecom, real-estate)
4. For each sector, compute sector factor = equal-weight mean of member stocks' log returns
5. Result: factor matrix (L, 1+K_sectors) per window — market + sector factors

#### Factor Generation Model
- Fit a VAR(p) model on the (1+K_sectors)-dimensional factor series
- Or train a tiny 1D diffusion model on factor sequences
- At sample time, first generate factor sequence, then condition InterDiff

#### Denoiser Changes
- Replace single `mkt_proj` with `factor_proj: Linear(1+K_sectors, d_model)`
- Each stock receives its sector's factor signal (not broadcast — use stock-to-sector mapping)

#### Data
- `build_dataset.py` outputs additional `sector_labels: (N_stocks,) int` array in the npz
- `panel_windows.py` computes per-sector factor from the sampled stocks + sector mapping

#### Verification
Same as Stage A, plus:
- Intra-sector correlation should be higher than inter-sector (currently not differentiated)
- Factor-regressed residual corr should remain close to real

## Files to Modify

### Stage A (minimal)
| File | Change |
|------|--------|
| `model.py` | Add `mkt_proj`, `mkt_cond` parameter to forward |
| `panel_windows.py` | Yield market factor as third element |
| `diffusion.py` | Pass `mkt_cond` through `training_loss` and `sample` |
| `train.py` | Unpack mkt_factor from batch, pass to model |
| `sample.py` | Bootstrap mkt_factor, pass to diffusion.sample |

### Stage B (after A)
| File | Change |
|------|--------|
| `build_dataset.py` | Add sector_labels to output npz |
| `panel_windows.py` | Compute sector factors, yield with batch |
| `model.py` | Expand mkt_proj to multi-factor proj, per-stock routing |
| New: `industry_map.py` | tushare code -> qlib code -> sector group mapping |

## Non-Goals
- Not changing the diffusion schedule or denoiser architecture (InterBlock structure stays)
- Not adding new evaluation metrics (existing stylized_facts + _diag_cs sufficient)
- Not training longer (20k steps is enough; the gap is architectural, not underfitting)
- Not touching Phase 0/1/3 code

## Risks
- Market factor conditioning could make the model lazy (just copy the factor, ignore stock-specific features) -> mitigate by keeping the factor as additive bias, not replacing stock input
- Bootstrap market factor limits diversity to what's in the training set -> acceptable for now, can switch to AR later
- Stage B sector grouping is somewhat arbitrary -> start with 10 groups, tune if needed
