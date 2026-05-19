你是一个学术文献分析师。把下面的 articles 按主题归类成 2-5 个 themes。

# 任务

读完所有 articles 的 title + tags + 一句话摘要，按相似度（同一方法 / 同一问题 / 同一数据集）聚类。

# Articles ({n_articles} 篇)

{articles_block}

# 输出严格 JSON

{{
  "themes": [
    {{
      "name": "<2-6 字的主题名，中文优先>",
      "description": "<1 句，说明这个主题的共同点>",
      "slugs": ["slug1", "slug2", "..."]
    }},
    ...
  ]
}}

# 规则

- 每个 article 必须出现在恰好一个 theme 里 —— 不可遗漏，不可重复
- themes 数量 2-5 个
- name 中文为主，技术词保留英文（"Diffusion 理论"、"对比实验"）
- 如果 articles 之间确实没什么共性，可以归一个大 theme 叫"杂"
