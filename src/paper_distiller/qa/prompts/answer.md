你正在为用户回答一个研究问题。所有相关的论文已经蒸馏好。

# 问题
{question}

# 可用的 articles（共 {n_articles} 篇）

{articles_full}

# 你的任务

综合所有 articles，写一个**完整、引用充分**的答案。

# 输出严格 JSON（无 markdown 围栏，无前导文字）

{{
  "title": "QA: <对问题的简短重述，中文>",
  "body": "<完整 markdown 答案>",
  "tags": ["...", "...", "3-7 个"],
  "cited_slugs": ["...实际在 body 里 [[link]] 引用的 slug..."]
}}

# body 结构要求

```
# QA: <title>

> **问题**: {question}

## 答案

[2-5 段，每段 200-400 字。引用 [[slug]] 形式 — slug 必须来自上面 articles 列表。]

## 关键发现要点

- 发现 1（来源: [[slug]]）
- 发现 2（来源: ...）
...

## 不确定 / 仍待研究

[列出还没回答的子问题。如果没有，就写"暂无明显未答的关键子问题"。]
```

约束：
- 答案的每个 claim 应有 [[link]] 支持。
- [[slug]] 必须来自上面 articles 列表 — **不要发明 slug**。
- 中文为主，技术词可保留英文。
- body 长度目标 800-2500 中文字。
