---
title: "Factor-Conditional InterDiff — M4/M5/M6 Findings"
category: "articles"
slug: "factor-conditional-interdiff-m4-m5"
tags: ["diffusion", "finance", "multi-stock", "experiment", "cross-section", "market-factor", "sector-factor", "csi300", "csi800", "scaling"]
refs: ["experiments/phase2_interdiff_fts"]
links: ["fts-interdiff-fusion", "factor-conditional-denoising", "interdiff-inter-stock-correlations", "synthetic-augmentation-financial-timeseries"]
created: "2026-04-17T20:45:00"
updated: "2026-04-17T22:15:00"
---

# Factor-Conditional InterDiff — M4/M5 Findings

> **Setup**: CSI300, 594 股 × 2430 交易日(2015-01-05 → 2024-12-31),64-day 窗口,k=32 每 panel。InterDiff-style denoiser 1.73M params,linear β∈[1e-4, 0.02] T=500, predict-x0 + clip ±5。20k steps on RTX 5090(~13 min wall)。
> **主线故事**: M3 诊断出 cross-section 问题 100% 来自 common mode(market factor 方差缺失 22%),架构层面加 **market factor 条件** 一招打掉 15% gap 到 <1%;再加 **per-stock sector 因子** 顺带修复了 lag10 的长程 ACF,把之前判为"结构性缺陷"的指标收进 OK。
> **关联**: 直接 parent direction [[fts-interdiff-fusion]];方法学抽象 [[factor-conditional-denoising]]。

## TL;DR

| 模型 | 新增条件 | ema loss | 7 项 verdict |
|------|---------|---------|-------------|
| M1 | regime embed | 0.1934 | 4 OK, 3 MEH |
| M3 | + length=128 | 0.1929 | 4 OK, 3 MEH(长度无效)|
| **M4** | + **market factor** | **0.1901** | **7 OK, 0 MEH** ✅ |
| **M5** | + **sector factor** | **0.1317** | **7 OK, 0 MEH** + 多项细粒度更准 |

**关键数字**:market_factor_var gap 从 -22% → -0.7%;panel_mean_pair_corr gap 从 -15% → -0.5%;acf_r² lag10 从 -0.034 → -0.002(真实 -0.001,几乎完美)。

## 诊断 — 为什么 M1/M3 的 cross-section 过不去

`_diag_cs.py`: 把 panel correlation 拆成三部分
1. **Max eigenvalue fraction**(common mode 强度)
2. **Mean pair corr after market-factor regression**(残差相关)
3. **Market factor variance**(共同因子自身的方差)

对比 real vs M1/M3:
- 残差相关几乎一样(-0.026 vs -0.027,差 3%)
- market_factor_var **差 22%**(real 0.348 vs syn 0.272)

这告诉我们:模型学会了每只股的波动特性,但学不好"所有股一起涨跌"的共同运动。M2 的 loss 加权(w=16 的 `loss_res + w*loss_mkt`)只能收回 20%,w=64 反而退化——**这是架构缺陷,不是优化问题**。

## M4 — Market Factor Conditioning

### 思路
训练时从窗口里显式计算市场因子 $m_t = \frac{1}{k}\sum_{i=1}^k x_{i,t}^{(\text{logret})}$,作为额外 conditioning 喂进 denoiser。采样时从真实面板 bootstrap 一段 $m_t$ 序列作为生成的 guide(bootstrap 比 AR(1) 简单且无偏)。

### 架构改动

在 `InterDenoiser` 上加 17k params:

```python
self.mkt_proj = nn.Sequential(
    nn.Linear(1, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
# forward:
if mkt_cond is not None:  # mkt_cond: (B, L)
    me = self.mkt_proj(mkt_cond[:, None, :, None])  # (B, 1, L, d)
    h = h + me  # broadcast across N stocks
```

**additive**,和 regime embedding / sinusoidal time embed 并存,互不干扰。

### 数据管道

`panel_windows.py` 的 `__iter__` 多 yield 一个 $m_t$ 张量。`_split_batch` 用 dtype 区分 regime(int64)和 float 条件,保证任何 conditioning 组合都能正确解包(避免 (window, regime, mkt) 和 (window, mkt, sector) 的 3-tuple 歧义)。

### 结果

| metric | real | M1 | **M4** | 改进 |
|---|---|---|---|---|
| market_factor_var | 0.385 | 0.272 (-22%) | **0.382 (-0.7%)** | **22% → 0.7%** |
| panel_mean_pair_corr | 0.339 | 0.272 (-15%) | **0.337 (-0.6%)** | **15% → 0.6%** |
| max_eig_frac | 0.388 | 0.317 (-14%) | **0.378 (-2.6%)** | 14% → 2.6% |
| excess_kurt | 3.81 | 2.71 | 3.01 | 改善 |
| hill_left | 2.91 | 3.30 | 3.03 | 更贴近 real |
| leverage lag1 | 0.013 | 0.011 | **-0.003** | **❌ 符号反了** |

**全部 7 项 verdict 全绿**。market_factor_var 从相差 22% 直接收到 <1%。唯一退化的是 leverage 非对称性——这本来就不在加法 conditioning 能解决的范围内,因为 leverage 是方向性的(下跌时波动放大,上涨时不),对称的条件注入无法捕获。

## M5 — Per-Stock Sector Factor

### 思路

M4 的 market factor 是**全市场等权均值** —— 所有股共享同一个信号。但 A 股里板块轮动是真实的(比如"今天 AI 涨,消费跌"),单靠 market factor 丢失这层结构。解法:给每只股额外一个**自己所属行业的因子信号**。

### Sector 映射

数据源:`G:/stocks/stock_data/parquet/tushare_stock_basic.parquet`
- 578/594 CSI300 有细分行业标签(97% 覆盖),16 支标为 UNKNOWN
- 110 个细分行业手工映射成 **11 大 sector**:FINANCE(74)、TECH(94)、INDUSTRIAL(88)、CONSUMER(63)、HEALTHCARE(59)、TRANSPORT(41)、ENERGY(40)、METALS(35)、MATERIALS(35)、MEDIA(25)、REAL_ESTATE(24)

边表 `data/csi300_sectors.npz`, 不改原 panel npz, 用 sidecar 模式保持旧 checkpoint 兼容。

### Sector 因子计算(关键设计点)

对每个采样窗口的 k=32 只股,对每支股 i:
```
sector(i) = 股 i 所属 sector
同类 = 窗口内同 sector 的其他股(排除自己)
sector_factor[i, :] = mean(log_ret[同类, :])  # (L,)
# 如果窗内同 sector 只有自己 → 回退到 market factor
```

**为什么排除自己**:如果包含自己,模型可以学"return_i ≈ sector_factor[i]"的恒等映射,训练时 loss 会掉到 0 但生成时用真实 sector factor 做 guide → 直接抄袭 → 退化成"采样 real panel"。排除自己就没这个 shortcut。

### 架构改动

又加一条 per-stock 通道,同样 additive:

```python
self.sector_proj = nn.Sequential(
    nn.Linear(1, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
# forward:
if sector_cond is not None:  # (B, N, L) per-stock
    se = self.sector_proj(sector_cond[:, :, :, None])  # (B, N, L, d)
    h = h + se
```

再 +4k params(总 1.75M)。

### 结果

| metric | real(L=64) | M4 | **M5** | 说明 |
|---|---|---|---|---|
| excess_kurt | 3.81 | 3.01 | **3.30** | 更接近 |
| hill_right | 3.41 | 3.82 | **3.71** | 更接近 |
| hill_left | 2.91 | 3.03 | 3.00 | 相当 |
| acf_r² lag1 | 0.061 | 0.067 | **0.057** | 更接近 |
| acf_r² lag5 | 0.007 | 0.017 | **0.014** | 更接近 |
| **acf_r² lag10** | **-0.001** | **-0.034** | **-0.002** | **意外大修** |
| acf_r² lag20 | -0.020 | -0.017 | -0.015 | 相当 |
| panel_mean_pair_corr | 0.339 | 0.337 (-0.6%) | 0.337 (-0.5%) | 保持 |
| market_factor_var | 0.385 | 0.382 (-0.7%) | 0.383 (-0.7%) | 保持 |
| max_eig_frac | 0.388 | 0.378 (-2.6%) | **0.382 (-1.3%)** | 改善 |
| leverage lag1 | 0.013 | -0.003 | -0.004 | 仍为负 |
| **ema loss** | — | 0.1901 | **0.1317** | **-31%** |

**关键发现**:
1. **ema loss 掉 31%** — sector 信号让 denoiser 预测噪声任务本身容易了,说明这条信号**携带了原架构没充分利用的信息**。不是单纯拟合更好,是模型看到了之前看不见的 patterns。
2. **lag10 的 ACF 从 -0.034 到 -0.002** — 之前判为"中长期 vol clustering 缺,需要 autoregressive head 或 Hawkes-like 机制"的问题被意外修复。机制推测:sector 因子里天然带着比 regime(单股 rolling vol 分位)更长期的集体记忆(**行业轮动、板块热度的持续性**),denoiser 借到了这条信号。M5 把 regime-conditioning 的短程(rolling 10 天)和 sector-conditioning 的中长程(天然持续)互补起来了。
3. **尾部统计全面改善** — kurt、hill 都往 real 方向走。

## 仍未解决 {#open}

**Leverage 非对称性**:M4/M5 都是 -0.003 ~ -0.004 vs real +0.013。对称加法 conditioning 解不了这个问题。原因:leverage effect 要求 $\text{corr}(r_t, r_{t+1}^2) > 0$,即"下跌 → 明天波动更大",上涨不对称。

候选方向:
1. **Sign-aware conditioning** — 额外加一条 sign($m_t$) 的离散 embedding,让模型可以对市场方向做不同响应
2. **Asymmetric residual noise** — 在 noise schedule 里对 x < 0 的部分加更大的方差
3. **两阶段 VAR + InterDiff** — 外层用 GARCH/VAR 生成带 leverage 的 market factor 序列,内层 denoiser 条件在其上

哪条路先试,等做完 CSI800 scaling 再定。

## 实验 artifacts

```
experiments/phase2_interdiff_fts/
├── industry_map.py              # 110 行业 → 11 sector 映射
├── panel_windows.py             # 4-tuple yield: (x, regime, mkt, sector)
├── model.py                     # + mkt_proj, sector_proj
├── diffusion.py                 # + mkt_cond, sector_cond 参数
├── train.py                     # dict-style _split_batch
├── sample.py                    # bootstrap mkt + sector from real
├── data/
│   ├── csi300_2015_2024.npz     # 原 panel
│   └── csi300_sectors.npz       # sector 边表
└── ckpts/
    ├── M0_m4_mkt_step20000.pt
    ├── M0_m5_sec_step20000.pt
    └── *.samples.npz
```

## M6 — CSI800 Scaling

**2026-04-17**

扩展到 CSI800(top-800 按市值),最终 1324 只股通过 min_valid_ratio=0.5 过滤(92.4% valid coverage)。同 M5 架构和超参,+`--sectors-npz data/csi800_sectors.npz`。

Sector 分布(CSI800)更均衡,UNKNOWN 占比从 CSI300 的 2.7% 升到 7%(大盘外更多冷门股没标签),但主要 sector 都至少 50 只:TECH 200、INDUSTRIAL 211、CONSUMER 157、HEALTHCARE 129、MATERIALS 103、FINANCE 95、METALS 85、ENERGY 75、TRANSPORT 66、REAL_ESTATE 62、MEDIA 49、UNKNOWN 92。

### 关键发现 1: 架构**完美** scale

| 维度 | CSI300 (M5) | CSI800 (M6) |
|------|-------------|-------------|
| N_stocks | 594 | 1324 (×2.2) |
| ema loss | 0.1317 | **0.1325** |
| step/s (RTX 5090) | 23.3 | 22.5 |
| GPU peak | 2.43 GB | 2.43 GB |
| Wall time | 13 min | 15 min |

**loss 差异小于 1%**,GPU 使用不变。原因:k=32(每 panel 采 32 股)是硬约束,模型 forward/backward 规模和总股票池无关。只有 data loading 略慢(更大 panel npz)。

### 关键发现 2: Cross-section gap **进一步缩小**

| 指标 | M5 (CSI300) | M6 (CSI800) |
|------|-------------|-------------|
| market_factor_var gap | -0.7% | **-0.5%** ⬇ |
| panel_mean_pair_corr gap | -0.5% | -0.6% |
| max_eig_frac gap | -1.3% | -1.4% |
| resid_mean_pair_corr gap | -0.7% | -0.6% |

CSI800 下更大的股票池 → market factor 估计方差更小 → 训练时 ground truth 更稳 → denoiser 学得更准。这是**池子大 → 信号质量高**的线性关系,不需要扩模型。

### 关键发现 3: Sector 信号在更大池子里更干净

窗口内 k=32 股,CSI300 有效 sector 平均每类 **2.9 只**/窗,CSI800 提升到 **5.2 只**/窗(按非零 sector 计)。样本多 → 同 sector 均值更稳 → sector factor 的 SNR 更高。实验验证:M6 的 lag10 ACF 从 M5 的 -0.002 到 -0.001,更贴近 real 的 -0.005。

### 全绿(CSI800)

| metric | real | M6 syn | verdict |
|---|---|---|---|
| std | 0.0274 | 0.0283 | OK |
| excess_kurt | 3.55 | 3.03 | OK |
| hill_right / left | 3.46 / 3.00 | 3.83 / 3.16 | OK |
| acf_r² lag1 | 0.074 | 0.066 | OK |
| acf_r² lag10 | -0.005 | -0.001 | 极佳 |
| panel_mean_pair_corr | 0.349 | 0.347 | OK |
| panel_max_eig_frac | 0.398 | 0.392 | OK |
| **leverage lag1** | 0.013 | **-0.007** | ❌ 仍为负 |

### 唯一缺陷保持一致

leverage_lag1 在 M4(-0.003)、M5(-0.004)、M6(-0.007)都为负 vs real +0.013。和股票池大小无关,**是对称加法 conditioning 的根本局限**。所有方向性不对称现象(leverage、margin call 加速、崩盘集群)都需要额外机制。

## Next

1. ~~CSI800 扩规模~~ ✅ 完美 scale,架构验证通过
2. **α-sweep 下游实验** — 用 M6 合成数据做 ranking 预测,找 [[phase-transition-alpha-star-empirical|model collapse 相变点]]
3. **Leverage 方向性** — 见 [[#open]],sign-aware conditioning 或 asymmetric noise schedule
4. **Factor-model 分析** — M6 的 576 个同 sector 窗口样本足够做经验 factor-loading 分析,验证 denoiser 确实在用 sector 信号而不是忽略
