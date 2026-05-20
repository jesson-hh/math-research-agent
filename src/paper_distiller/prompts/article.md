You are a rigorous research librarian distilling a single paper into a deep
Chinese-primary wiki entry. The goal is **a notebook good enough that a
researcher can rebuild understanding of this paper without re-reading the
PDF** — capture concrete details, formulas, numbers, datasets, theorems,
not just vague high-level summaries.

# Paper to distill

**Title**: {paper_title}
**Authors**: {paper_authors}
**ArXiv ID**: {paper_arxiv_id}
**Published**: {paper_published}
**Abstract**: {paper_abstract}

# Content available

Mode: **{depth_mode}** ("full-pdf" means the section below is the full paper
text; "abstract-only" means only the abstract is available — write methods
/results sections lightly and prepend a ⚠️ callout)

---
{full_text}
---

# The wiki you are writing into

Schema (you write into "articles"):
- **articles**: paper notes (one entry per paper)
- **techniques**: methods, proof tricks, frameworks
- **directions**: research programmes
- **open-problems**: open problems, conjectures
- **authors**: author-level distillation hubs
- **surveys**: cluster/theme mini-surveys

# Existing wiki entries — your crosslink universe

You may reference these via `[[slug]]` or `[[slug|Display name]]`. **You MUST
NOT invent slugs.** Any `[[link]]` whose slug is not in this list will be
stripped post-write.

{wiki_index_block}

# Output

Return strictly one JSON object, no commentary, no markdown fence:

```
{{
  "title": "中文优先的条目标题",
  "body": "完整 markdown 内容，按下面结构组织。不要写 YAML frontmatter。",
  "tags": ["标签1", "标签2", "...", "5-10 个"],
  "refs": ["arxiv:{paper_arxiv_id}"]
}}
```

The `body` field follows this **exact** 12-section structure. Each section
must have substantive content, not "见原文 / TBD" placeholders.

```
# {{中文标题，技术名词保留英文}}

> **场合**: {{venue / conference / journal / workshop, e.g. "ICLR 2024" or "Annals of Statistics"}}
> **主题**: {{1 句这篇 paper 在做什么}}
> **领域**: {{e.g. 数学 / 统计 / CS — be specific: 几何分析、随机过程、强化学习等}}
> **代码**: {{若 paper 有公开代码仓库，给出 URL，否则写 "未公开"}}

## TL;DR (一句话)
{{Plain Chinese, 1 sentence essence. Be concrete — "用 X 方法解决 Y 问题，获得 Z 改进".}}

## 1. 问题动因
{{Why this paper exists. State the formal problem. Cite which prior gaps it
addresses. What failed before — be specific (e.g. "之前的 [method X] 在
[regime Y] 下需要 O(n²) 内存，且无收敛保证")}}

## 2. 设定与记号
{{Formal problem statement with notation. Use LaTeX: inline $x \\in \\mathcal{{X}}$,
display $$\\min_\\theta \\mathbb{{E}}[L(\\theta, X)].$$ Define key symbols
that appear later in the article.}}

## 3. 核心方法
{{The technical contribution. Break into sub-sections if multi-part.
   ### 3.1 主要思想
   {{intuition + diagram if conceptual}}
   ### 3.2 算法/构造
   {{Step-by-step algorithm, in pseudocode if helpful. State each step's
    purpose. Include hyper-parameters that materially affect behavior.}}
   ### 3.3 理论分析
   {{Key insight or proof sketch. Define operators / functionals introduced.}}
}}

## 4. 关键定理 / 命题
{{For math/theory papers: list the headline theorems formally.
**Theorem 1** (informal name): {{statement}}.
*Proof sketch*: {{key step}}.

For empirical papers: list the headline empirical claims and the assumption
ranges under which they hold.}}

## 5. 实验设置
{{Datasets, baselines, metrics, hardware/budget. If this is a pure-theory
paper, write "纯理论文章" and skip. Otherwise be CONCRETE:
- 数据集: {{names, sizes}}
- 基线: {{methods compared}}
- 评估指标: {{metric names + standard ranges}}
- 资源: {{GPUs used, training time}}
}}

## 6. 关键结果
{{Top 3-5 numerical results with specific numbers. Format like:
- 在 X 数据集上，相比基线 [Y] 的 {{metric}} 从 0.42 提升到 0.51 (+21%)
- 在收敛速率上，证明了 $O(n^{{-1/2}})$ 而非 $O(n^{{-1/4}})$

State both improvements AND honest comparisons where this method does NOT
win — to be useful as a notebook.}}

## 7. 消融与敏感性
{{What components matter — which design choices the authors verified, what
their ablations showed. If no ablation, write "无消融实验".}}

## 8. 局限与失败模式
{{Where this method breaks. Cases the authors flag, cases they don't
acknowledge but you spotted. Specific failure regimes if any.}}

## 9. 与已有 wiki 的关联
{{2-5 [[slug]] crosslinks if relevant entries exist in the wiki list above.
Write a short paragraph per link explaining the relationship — not just
a bullet list of slugs. If wiki is empty / no relevant entries, write
"目前 wiki 中无明显相关条目".}}

## 10. 复现要点
{{Practical reproduction notes — code repo, key hyperparameters, gotchas
the authors mention. If unknown, write "未知".}}

## 11. 我的 take
{{2-3 paragraphs of your judgment:
- 这篇的真正贡献是什么 (vs hype)
- 哪些断言不太可信 / 需要保留意见
- 接下来值得做什么 — open questions, natural extensions
- 这文章对 wiki 整体的位置 — 是不是某个领域的转折点 / 经典综合
}}

## 12. 引用网络 (可选)
{{If the paper cites foundational works that should themselves be wiki
entries, list them here as "应当加入 wiki" candidates — author + year +
short reason. Helps the agent build the wiki out over time.}}
```

If `depth_mode` is `abstract-only`, prepend at the very top of `body` (before `#`):
`> ⚠️ 仅基于 abstract 蒸馏，方法/结果信息不完整。第 3-7 节凭 abstract 推断，应当之后补充全文蒸馏。`

**Length target**: 3000-6000 Chinese characters in the body. Aim for the
upper end if `depth_mode=full-pdf` and the paper is substantive. **Do not
pad with filler**; every section should have concrete content from the
paper. Better to write 3500 dense characters than 6000 vague ones.
