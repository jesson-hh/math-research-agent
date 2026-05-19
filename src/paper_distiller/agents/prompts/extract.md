你是一个学术文献分析师。从下面的 article markdown 中提取结构化信息。

# 任务

读完整个 article，抽出四类内容：

1. **theorems** — 论文中正式陈述的定理（含数学结论的简短描述）
2. **assumptions** — 关键假设（如 "Lipschitz score"、"compact support"、"bounded variance"）
3. **convergence_rates** — 任何收敛速率（如 "O(n^{{-1/d}})"、"O(n^{{-1/(2β+d)}})"）
4. **key_lemmas** — 重要引理 / 中间结果

# Article ({slug})

{article_body}

# 输出严格 JSON

{{
  "theorems": ["..."],
  "assumptions": ["..."],
  "convergence_rates": ["..."],
  "key_lemmas": ["..."]
}}

# 规则

- 每项保持简短（~30 字以内）
- 没有就给空 list `[]`
- LaTeX 公式保留 `$...$` 包裹
- 中英都行，技术词保留原文
