---
title: "Factor-Conditional Denoising for Multi-Stock Diffusion"
category: "techniques"
slug: "factor-conditional-denoising"
tags: ["diffusion", "conditioning", "factor-model", "film", "multi-stock", "method"]
refs: ["experiments/phase2_interdiff_fts"]
links: ["fts-interdiff-fusion", "factor-conditional-interdiff-m4-m5", "interdiff-inter-stock-correlations"]
created: "2026-04-17T20:50:00"
updated: "2026-04-17T20:50:00"
---

# Factor-Conditional Denoising

> **One-line**: 在多股 diffusion denoiser 里加一条(或几条)**低维因子信号**作为加法 conditioning,显式承载共同运动(market)和子集共同运动(sector/industry)。
> **验证案例**: [[factor-conditional-interdiff-m4-m5]] 把 InterDiff 的 cross-section gap 从 15% 收到 <1%。
> **上下文**: 适用于任何有"共同因子 + 特质残差"结构的 diffusion 生成(金融 panel、ensemble weather、神经群体活动)。

## 问题

多股(multi-stock)diffusion 典型缺陷:denoiser 内部的 inter-stock attention **低估共同运动方差**。诊断方式(`_diag_cs` 风格):
$$
\text{mean\_pair\_corr} = \text{top\_eig\_frac}(C) + \text{residual\_corr}
$$
其中 $C$ 是 panel 相关阵。把 synthetic 和 real 的三项各自算出来对比:
- 若 residual_corr 接近 real、top_eig_frac 偏低 → **common mode 是根因**
- 若 residual_corr 也偏低 → 残差结构也学得不好,需要改架构(不是本技术能解决的)

金融面板(A 股)上经验表明,残差部分 inter-attention 学得很好,**几乎全部 gap 都来自 common mode**。

## 方法

### 1. 因子提取(训练时)

对每个训练窗口 $(x_{i,t})_{i=1..k, t=1..L}$(k 只股 × L 时间步 × C 通道,第 0 通道是 log return):
- **Market factor**: $m_t = \frac{1}{k}\sum_{i=1}^k x_{i,t}^{(0)}$,shape $(L,)$
- **Per-stock sector factor**: 股 i 所属 sector $s(i)$,则
  $$s_{i,t} = \frac{1}{|\{j \ne i : s(j) = s(i)\}|} \sum_{j \ne i, s(j) = s(i)} x_{j,t}^{(0)}$$
  shape $(k, L)$。**关键**:排除 self 避免 denoiser 学恒等映射 shortcut。

如果窗内同 sector 只有自己,$s_{i,t}$ 回退到 $m_t$。

### 2. 因子注入(architectural)

**加法 FiLM-style**。每条因子独立投影到 $d_{\text{model}}$ 后加到 denoiser 的 hidden state:

```python
# market factor: (B, L) -> broadcast (B, 1, L, d_model)
self.mkt_proj = Sequential(
    Linear(1, d_model), GELU(), Linear(d_model, d_model)
)
# per-stock sector: (B, N, L) -> (B, N, L, d_model)
self.sector_proj = Sequential(
    Linear(1, d_model), GELU(), Linear(d_model, d_model)
)

# forward (加法,和 time/regime embedding 并存):
if mkt_cond is not None:
    h = h + self.mkt_proj(mkt_cond[:, None, :, None])
if sector_cond is not None:
    h = h + self.sector_proj(sector_cond[:, :, :, None])
```

**为什么用加法 + 独立 MLP 而不是 cross-attention**:
- 因子信号是**连续标量**,每时间步一个值,没有"token 序列"结构,cross-attention 是过杀
- 加法和现有的 regime embed / time embed 同构,训练稳定性好
- 参数开销极小(每条因子 ~17k params,占总模型 <1%)

**为什么 sector 用 per-stock 信号而不是 broadcast**:
- 每股只"看见"自己所属 sector 的因子,物理意义对
- 如果把所有 K 个 sector 因子都喂给每个 stock token,模型要学 routing(谁关注哪个),浪费容量
- 代价:$(B, N, L)$ 比 $(B, L)$ 多 N 倍内存,但因为后面的 attention 层本来就是 $(B, N, L, d)$ 规模,加到其上无额外开销

### 3. 因子生成(采样时)

采样时需要 $m_t$ 和 $s_{i,t}$ 序列作为 conditioning,但我们没有真实的——要生成。两种策略:

**(a) Bootstrap**(起步推荐):随机从真实 panel 里取一个 $(k, L)$ 窗口,把对应的 factor 序列抽出来用作 guide。简单、无偏、计算量 0。

**(b) 参数化模型**:对真实 factor 序列训一个 AR(1) / VAR / 小型 1D diffusion,采样时从中生成新序列。更灵活(可以生成真实没见过的 regime),但要额外训练和调参。

Bootstrap 的缺点:生成的 panel 被绑定到真实 factor 路径,novelty 受限。对数据增强 OK,对"合成新市场情景"不够。先 bootstrap 跑通再升级。

### 4. 训练/采样一致性

**训练**:每个 mini-batch 里,真实窗口 → 算真实 factor → 喂进 denoiser 预测噪声。
**采样**:先 bootstrap factor → 条件上做 reverse diffusion → 得到 panel。

两端都用真实 factor 分布,没有 train/test gap。

## 为什么 loss 加权(M2)不行

我们试过的替代方案:把 loss 拆成 `loss_res + w * loss_mkt`,让模型显式关注 market factor 预测。结论:
- w=1 数值恒等 base_MSE(已验证),无效
- w=16 最优但只收回 20% gap
- w>16 整体 landscape 扭曲,采样质量反而退化

**根因**:loss 加权只改变训练信号的权重,**没改变 denoiser 的 inductive bias**。denoiser 如果看不到 market factor 作为输入,它必须从 noised panel 自己"猜"出来,而噪声把共同模式淹没了——这是架构问题,只能从架构层面解决。

## 泛化

这个技术本质是:**把已知的低维结构显式 inject 进生成模型的 hidden state,而不是让模型从纯噪声里推断出来**。应用模式:
- 金融 panel: market + sector(本文)
- 天气 ensemble: 全球平均温度 + 区域平均
- 神经活动: 全脑均值 + 各皮层区域均值
- 图像:低通滤波 + bandpass 各频段

通用配方:
1. 分解数据 = 共同模式 + 特质残差
2. 生成时:先生成共同模式(简单)→ 条件生成残差(难但 denoiser 擅长)
3. 架构上:加法 FiLM 注入 + 独立小 MLP 投影

## 局限

- **对称加法 conditioning 无法建模非对称现象**,比如金融里的 leverage effect(下跌放大波动但上涨不)。需要 sign-aware conditioning 或 asymmetric noise schedule。
- **Bootstrap factor 的多样性受限** 训练集 factor 分布。生成"真实没见过的新 regime"需要参数化 factor model。
- **Sector 映射是超参数**:Sector 数量、细粒度、成员归属,都会影响 sector factor 的信噪比。本文用 11 类是经验值,最优粒度未调。

## 关联

- [[fts-interdiff-fusion]]: 这个技术的 host 方向
- [[factor-conditional-interdiff-m4-m5]]: 具体实验结果和数字
- [[interdiff-inter-stock-correlations]]: 原始 InterDiff 架构
- [[synthetic-augmentation-financial-timeseries]]: 下游应用目标
