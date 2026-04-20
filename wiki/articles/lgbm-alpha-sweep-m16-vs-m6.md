---
title: "LGBM α-sweep M6 vs M16: Better Stylized Facts ≠ Better Augmentation"
category: "articles"
slug: "lgbm-alpha-sweep-m16-vs-m6"
tags: ["alpha-sweep", "lightgbm", "auxiliary-loss", "regularization", "mode-collapse", "negative-result", "csi800"]
refs: ["experiments/phase2_interdiff_fts/lgbm_sweep.py"]
links: ["m10-m17-leverage-engineering", "lgbm-alpha-sweep-phase-transition", "factor-conditional-interdiff-m4-m5", "fts-interdiff-fusion"]
created: "2026-04-20T00:00:00"
updated: "2026-04-20T00:00:00"
---

# LGBM α-Sweep: M16 vs M6 — A Negative Result

> **TL;DR**: M16 has better stylized facts than M6 (leverage -0.0037 vs -0.007, skew +0.118 vs -0.099 near real +0.106, max_eig perfect match vs -1.3%) but **delivers LESS downstream IC at any α**. Fair 10-seed LGBM sweep shows M6 α=0.5 gives IC paired Δ = +0.0087 (t=1.68), while M16 at same α gives Δ = +0.0012 (t=0.36, not significant). **Stylized-fact improvements from aux-loss regularization came at the cost of augmentation value.**
> **Relation**: Sanity check of [[m10-m17-leverage-engineering]] against [[lgbm-alpha-sweep-phase-transition]].

## Setup

Same lgbm_sweep.py as the original M6 α-sweep, but run with:
- 10 seeds (instead of 5 previously — to tighten CI)
- Two synth sources: M6 samples vs M16 samples (M16 = M6 + t-EDM + hinge leverage aux)
- Same LGBM config, 100k training rows, 2023 test

Makes M6 and M16 directly comparable as drop-in replacements.

## Head-to-head results (10 seeds each)

| α | **M6 IC** | **M16 IC** | M6 paired t vs α=0 | M16 paired t vs α=0 |
|---|---|---|---|---|
| 0.00 | +0.00212 | +0.00212 | — | — |
| 0.10 | +0.00147 | +0.00221 | -0.20 | +0.03 |
| 0.25 | +0.00436 | +0.00358 | +0.61 | +0.33 |
| **0.50** | **+0.00873** | **+0.00329** | **+1.68** | +0.36 |
| 0.75 | +0.00226 | +0.00353 | +0.04 | +0.27 |
| 0.90 | +0.00484 | +0.00121 | +0.56 | -0.20 |

### Per-seed paired difference (M16 - M6)

| α | Δ = M16_IC - M6_IC | std | t-stat |
|---|---|---|---|
| 0.00 | 0 | 0 | — |
| 0.25 | -0.00079 | 0.012 | -0.21 |
| **0.50** | **-0.00544** | 0.011 | **-1.55** |
| 0.75 | +0.00126 | 0.010 | +0.39 |
| 0.90 | -0.00363 | 0.010 | -1.15 |

**At α=0.5, M16 is worse than M6 by 0.0054 IC, approaching significance (t=-1.55)**.

## Stylized-fact quality comparison

| metric (vs real) | M6 | M16 | "Quality" winner |
|---|---|---|---|
| leverage_lag1 | -0.007 | **-0.0037** | M16 ✓ |
| skew | -0.099 | **+0.118** (real +0.106) | M16 ✓ |
| hill_right | 3.85 | **3.57** (real 3.46) | M16 ✓ |
| excess_kurt | 3.03 | **3.15** (real 3.55) | M16 ✓ |
| pair_corr | 0.347 | 0.353 (real 0.349) | ≈ |
| max_eig_frac | 0.394 | **0.398** (real 0.398) | M16 ✓ |
| acf_r²_lag1 | 0.066 | 0.058 (real 0.074) | M6 ✓ (marginal) |

M16 wins on 5/7 stylized-fact indicators, ties on 1, loses marginal on 1.

**But M6 wins decisively on downstream IC at α=0.5**. Surface quality ≠ utility.

## Three hypotheses for this disconnect

### H1: Mode collapse by regularization

The hinge aux loss is a global constraint: `leverage(x0_pred) ≥ +0.013`. Satisfying this at every batch biases the model toward x0_pred outputs with specific structure. The output manifold effectively shrinks — less of the true conditional diversity is preserved.

Augmentation value depends on *extra variation* that the predictor can learn from. A narrower generation manifold = fewer useful extra examples.

### H2: Push toward "average" via aux statistic

The aux loss depends on a **batch-pooled statistic** (correlation over all trajectories). The cheapest way for the optimizer to increase batch-pooled leverage is to make *all* trajectories behave more similarly — reducing per-trajectory idiosyncrasies. LGBM relies on idiosyncratic features for splits, so loses signal.

### H3: Leverage irrelevant to this predictor

LGBM's features (17 hand-engineered: momentum, realized vol, ranges, skew/kurt, cross-sectional rank) might not load on the leverage signal. In that case fixing leverage gives no uplift, and the extra aux-loss regularization is pure cost.

H1 and H2 both predict M16 < M6 regardless of predictor. H3 predicts M16 ≈ M6 (no change either way). The observed M16 **worse than** M6 is most consistent with **H1+H2 dominating**, with leverage-relevance (H3) possibly contributing but not the main driver.

## Methodological implications

### Stylized-fact verdicts are not a utility proxy

Our eval_compare 7-metric verdict catches coarse failures but not generation diversity or "usefulness for augmentation". A model can score 7/7 OK with great margins, yet produce less useful training data.

**Need new evaluation**: some measure of sample diversity / coverage beyond marginal statistics. Candidates:
- Nearest-neighbor distance distribution (typicality)
- MMD vs real on joint statistics
- Diversity metric: average pairwise distance in feature space
- Or just: always include downstream IC α-sweep in the evaluation pipeline

### Aux loss trades surface for depth

Explicit statistical constraints (like leverage target) are "surface-level". They say *what* the distribution should look like on one metric. But they don't constrain *the generation mechanism*, which is what determines utility.

For downstream usefulness, unconstrained optimization with strong factor conditioning (M6) seems to beat strongly-constrained optimization tailored to specific stylized facts (M16).

### When would M16 be preferred?

- **Risk models / VaR**: need tail accuracy, M16's kurt/hill better
- **Stress testing**: need realistic leverage, M16's (partial fix) better
- **Public distribution / paper figures**: need visual match to real stylized facts

But NOT for:
- **Downstream data augmentation**: M6 better
- **Generic synthetic data needs**: M6 better

## Recommendation

**Return to M6 as the default production model** for downstream ML use cases.

Keep M16 available as an alternative for surface-quality-sensitive applications (risk, stress, publication). Document both.

If sign flip on leverage is ever required:
- Don't use aux loss (it costs too much downstream)
- Try per-trajectory aux (may be less diversity-destructive) — untested
- Or GJR-GARCH 2-stage (theoretical clean fix; factor trajectories with right leverage sign fed as conditioning to an *unconstrained* M6-style denoiser) — most promising

## Artifacts

```
experiments/phase2_interdiff_fts/ckpts/
├── alpha_sweep_m6_lgbm_10seeds.json    (new, apples-to-apples)
├── alpha_sweep_m16_lgbm_10seeds.json   (new)
└── alpha_sweep_lgbm.json               (original 5-seed M6)
```

## Takeaway

> **"Better stylized facts" is a necessary but not sufficient condition for better generative models.** Diversity and generation-mechanism fidelity also matter, and our current evaluation pipeline doesn't capture them.
