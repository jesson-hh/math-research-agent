// paper-distiller — interactive prototype
const { useState, useEffect, useRef, useCallback } = React;

// ───────────────────────────────────────────────────────────
// DATA

const ARTICLE = {
  arxiv: "1706.03762",
  title: "Attention Is All You Need",
  authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
  venue: "NeurIPS 2017",
  tags: ["transformer", "attention", "scaled-dot-product", "multi-head", "seq2seq"],
  sections: [
    {
      num: "§ 2",
      ref: "p. 1",
      title: "TL;DR",
      body: [
        { type: "p", text: "抛弃 RNN/CNN,只用注意力机制构建编码器–解码器,在两项 WMT 翻译任务上以更少的训练时间取得 SOTA。Transformer 由此成为后续大语言模型的基石。" }
      ]
    },
    {
      num: "§ 5",
      ref: "p. 3-4",
      title: "核心方法 · Scaled dot-product attention",
      body: [
        { type: "p", text: "给定 $Q$、$K$、$V$ 三组向量,attention 把 query 与所有 key 做点积、按 $\\sqrt{d_k}$ 缩放、过 softmax,得到一组权重对 value 加权求和:" },
        { type: "math", tex: "\\mathrm{Attention}(Q,K,V) \\;=\\; \\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V", tag: "1" },
        { type: "p", text: "缩放因子 $\\sqrt{d_k}$ 用来稳定 softmax 在 $d_k$ 较大时的梯度 — 假设 $Q,K$ 各分量 i.i.d. 单位方差,则 $QK^{\\top}$ 的方差为 $d_k$,除以 $\\sqrt{d_k}$ 把方差压回 $O(1)$。" }
      ]
    },
    {
      num: "§ 6",
      ref: "p. 4-5",
      title: "关键定理 · Multi-head 表达能力",
      body: [
        { type: "p", text: "用 $h$ 个独立投影并行做注意力,等价于在不同子空间观察序列。当 $h \\cdot d_k = d_{\\text{model}}$ 时,multi-head 与单头同参数量,但实证表达力更强。" },
        { type: "math", tex: "\\mathrm{MultiHead}(Q,K,V) \\;=\\; \\mathrm{Concat}(\\mathrm{head}_1,\\ldots,\\mathrm{head}_h)\\,W^{O}", tag: "2" },
        { type: "math", tex: "\\text{where}\\quad \\mathrm{head}_i \\;=\\; \\mathrm{Attention}(QW_i^{Q},\\,KW_i^{K},\\,VW_i^{V})", tag: null, small: true }
      ]
    }
  ]
};

const SEARCH_RESULTS = [
  { idx: "①", title: "Attention Is All You Need", meta: "Vaswani et al · 2017", arxiv: "1706.03762" },
  { idx: "②", title: "Layer Normalization", meta: "Ba et al · 2016", arxiv: "1607.06450" },
  { idx: "③", title: "Transformer-XL", meta: "Dai et al · 2019", arxiv: "1901.02860" },
  { idx: "④", title: "On the Variance of the Adaptive Learning Rate", meta: "Liu et al · 2019", arxiv: "1908.03265" },
  { idx: "⑤", title: "FlashAttention: Fast and Memory-Efficient Exact Attention", meta: "Dao et al · 2022", arxiv: "2205.14135" }
];

const SEED_PROMPTS = [
  { tag: "蒸馏", text: "找几篇 Transformer 注意力机制的核心论文,蒸馏 3 篇" },
  { tag: "问答", text: "解释一下 Attention Is All You Need 里 √d_k 的作用" },
  { tag: "审查", text: "把开放的这篇证明跑一遍审查,找最值得手工核对的步骤" },
  { tag: "深度研究", text: "亚二次复杂度的 attention 路线有哪些?" }
];

const RECENT_ARTICLES = [
  { title: "Layer Normalization", meta: "1607.06450 · 2 hours ago" },
  { title: "FlashAttention v2", meta: "2307.08691 · yesterday" },
  { title: "Mamba: Linear-Time Sequence Modeling", meta: "2312.00752 · 3 days ago" },
  { title: "S4: Efficient Long Sequences", meta: "2111.00396 · 1 week ago" }
];

// ───────────────────────────────────────────────────────────
// HELPERS

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const uid = () => Math.random().toString(36).slice(2, 9);

// ───────────────────────────────────────────────────────────
// MATH — KaTeX-backed

// Renders TeX. `display` = block-mode (centered, larger). Use TeX inside JSX freely.
function TeX({ tex, display = false }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const render = () => {
      if (!window.katex) { setTimeout(render, 30); return; }
      try {
        window.katex.render(tex, ref.current, {
          displayMode: display,
          throwOnError: false,
          strict: "ignore",
          output: "html"
        });
      } catch (e) {
        ref.current.textContent = tex;
      }
    };
    render();
  }, [tex, display]);
  return <span ref={ref} className={display ? "tex-display" : "tex-inline"} />;
}

// Display equation block — rule bar + (optional) tag.
function Equation({ tex, tag, small }) {
  return (
    <div className={"eq-block" + (small ? " eq-small" : "")}>
      <span className="eq-rule"></span>
      <div className="eq-body">
        <TeX tex={tex} display />
      </div>
      {tag && <span className="eq-tag">({tag})</span>}
    </div>
  );
}

// Renders a CJK string with `$...$` segments rendered as inline KaTeX.
function RichText({ children }) {
  const str = String(children);
  const out = [];
  let i = 0;
  let key = 0;
  while (i < str.length) {
    if (str[i] === "$") {
      // find closing $
      const end = str.indexOf("$", i + 1);
      if (end > i) {
        out.push(<TeX key={key++} tex={str.slice(i + 1, end)} />);
        i = end + 1;
        continue;
      }
    }
    // text run until next $
    const next = str.indexOf("$", i);
    const stop = next === -1 ? str.length : next;
    out.push(<React.Fragment key={key++}>{str.slice(i, stop)}</React.Fragment>);
    i = stop;
  }
  return <>{out}</>;
}

// ───────────────────────────────────────────────────────────
// TOPBAR

function TopBar({ cost, busy, dark, setDark }) {
  return (
    <header className="topbar">
      <div className="brand">paper—<em>distiller</em></div>
      <div className="topbar-right">
        <div className={"cost-chip" + (busy ? " busy" : "")}>
          <span className="dot"></span>
          <span>{busy ? "running" : "ready"}</span>
          <span style={{ opacity: 0.35 }}>·</span>
          <span>¥{cost.toFixed(2)}</span>
        </div>
        <button className="icon-btn" onClick={() => setDark(d => !d)} title="theme">
          {dark ? "☀" : "◐"}
        </button>
      </div>
    </header>
  );
}

// ───────────────────────────────────────────────────────────
// CHAT — empty state with seed prompts

function EmptyChat({ onSeed }) {
  return (
    <div className="empty">
      <h1>What are we <em>reading</em> today?</h1>
      <p>用自然语言告诉它你想读什么。它会去 arXiv 找论文、写中文笔记、建证明图,并把所有材料沉淀到本地 vault。</p>
      <div className="seeds">
        {SEED_PROMPTS.map((p, i) => (
          <button key={i} className="seed" onClick={() => onSeed(p.text)}>
            <span className="seed-tag">{p.tag}</span>
            {p.text}
          </button>
        ))}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// MESSAGES

function UserMsg({ text }) {
  return <div className="msg-user">{text}</div>;
}

function AsstMsg({ text, streaming, children }) {
  return (
    <div className="msg-asst">
      {text}
      {streaming && <span className="caret"></span>}
      {children}
    </div>
  );
}

function ToolCard({ name, status, args, results, onPick }) {
  const statusText = {
    running: "running",
    done: results ? `done · ${results.length} 条 · ¥0.001` : "done · ¥0.001"
  }[status];
  return (
    <div className="tool">
      <div className="tool-head">
        <div className="tool-name">{name}</div>
        <div className={"tool-status " + status}>{statusText}</div>
      </div>
      {args && (
        <div className="tool-body">
          {Object.entries(args).map(([k, v]) => (
            <div key={k} className="tool-arg">
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ))}
        </div>
      )}
      {results && (
        <div className="tool-result">
          {results.map((r, i) => (
            <div
              key={i}
              className={"tool-result-row" + (onPick ? " clickable" : "")}
              onClick={onPick ? () => onPick(r) : undefined}
            >
              <span className="idx">{r.idx}</span>
              <span>{r.title}</span>
              <span className="meta">{r.meta}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// INPUT

function InputBox({ value, setValue, onSend, disabled }) {
  const ref = useRef(null);
  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = Math.min(ref.current.scrollHeight, 160) + "px";
    }
  }, [value]);
  return (
    <div className="input-wrap">
      <div className="input-box">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={disabled ? "agent 正在工作中…" : "键入消息,或按 / 触发斜杠命令…"}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          disabled={disabled}
        />
        <button
          className="input-send"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          title="send"
        >↗</button>
      </div>
      <div className="input-meta">
        <span>⏎ 发送 · ⇧⏎ 换行 · / 命令</span>
        <span>auto mode · qwen-plus</span>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// WORKSPACE — welcome view

function WelcomeView({ onPick }) {
  return (
    <div className="welcome-wrap">
      <div className="greeting">YOUR VAULT · ~/research</div>
      <h2>欢迎回来。<br /><em>已经读过</em>这些论文。</h2>
      <p className="lead">左边问点什么开始,或者直接打开一篇最近的笔记。所有内容都存在本地 — 可以用 Obsidian 打开。</p>

      <div className="stat-grid">
        <div className="stat">
          <div className="stat-k">articles</div>
          <div className="stat-v"><em>242</em></div>
        </div>
        <div className="stat">
          <div className="stat-k">surveys</div>
          <div className="stat-v"><em>14</em></div>
        </div>
        <div className="stat">
          <div className="stat-k">proof nodes</div>
          <div className="stat-v"><em>3.6k</em></div>
        </div>
      </div>

      <div className="recent-h">Recent</div>
      <div className="recent-list">
        {RECENT_ARTICLES.map((r, i) => (
          <div key={i} className="recent-item" onClick={onPick}>
            <span className="recent-title">{r.title}</span>
            <span className="recent-meta">{r.meta}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// WORKSPACE — article view

function ArticleView({ article, articleFlash, onOpenGraph, onOpenPaper }) {
  return (
    <div className="article-wrap">
      <div className="article-eyebrow">
        <span>ARTICLE · DISTILLED</span>
        <span className="arxiv">{article.arxiv}</span>
        <button className="eyebrow-cta" onClick={() => onOpenPaper()}>
          <span>↗</span> view original
        </button>
      </div>
      <h1>{article.title}</h1>
      <p className="article-authors">{article.authors} · {article.venue}</p>
      <div className="article-tags">
        {article.tags.map(t => <span key={t} className="tag">{t}</span>)}
      </div>

      {article.sections.map((s, i) => (
        <div key={i} data-section={s.num} className={"article-section" + (articleFlash === s.num ? " sec-flash" : "")}>
          <h3>
            <button className="num" onClick={() => onOpenPaper(s.num)} title={`原文 ${s.ref}`}>
              {s.num}
            </button>
            <span>{s.title}</span>
            <button className="section-ref" onClick={() => onOpenPaper(s.num)}>
              {s.ref} ↗
            </button>
          </h3>
          {s.body.map((b, j) => {
            if (b.type === "p") return <p key={j}><RichText>{b.text}</RichText></p>;
            if (b.type === "math") return <Equation key={j} tex={b.tex} tag={b.tag} small={b.small} />;
            return null;
          })}
        </div>
      ))}

      <div className="notice" style={{ marginTop: 32 }}>
        <span className="nicon">⌗</span>
        <p>
          这篇有 <span style={{ color: "var(--accent)" }}>23 个证明节点</span>,
          其中 <span style={{ color: "var(--warn)" }}>3 个 review 时被标可疑</span>。
          <span className="wiki" onClick={onOpenGraph} style={{ color: "var(--accent-2)", borderBottom: "1px solid var(--accent-2)", cursor: "pointer", marginLeft: 6 }}>打开证明图谱 →</span>
        </p>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// PAPER VIEW — original PDF-like layout

const PAPER_FOOTNOTES = {
  "4": "To illustrate why the dot products get large, assume that the components of Q and K are independent random variables with mean 0 and variance 1. Their dot product Q·Kᵀ = Σ q_i k_i has mean 0 and variance d_k."
};

function PaperView({ jumpSection, jumpStamp }) {
  const [page, setPage] = useState(3);
  const [zoom, setZoom] = useState(100);
  const [flashId, setFlashId] = useState(null);
  const [fnOpen, setFnOpen] = useState(null);
  const pageRef = useRef(null);
  const scrollRef = useRef(null);

  // Map article section refs to (page, anchor)
  const SEC_MAP = {
    "§ 2": { page: 1, anchor: null },
    "§ 5": { page: 3, anchor: "sec-3-2-1" },
    "§ 6": { page: 4, anchor: "sec-3-2-2" }
  };

  useEffect(() => {
    if (!jumpSection) return;
    const target = SEC_MAP[jumpSection];
    if (!target) return;
    setPage(target.page);
    // wait for render, then smooth-scroll within the paper container
    const t1 = setTimeout(() => {
      if (target.anchor) {
        const el = document.getElementById(target.anchor);
        const container = scrollRef.current;
        if (el && container) {
          const top = el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop - 80;
          container.scrollTo({ top, behavior: "smooth" });
          setFlashId(target.anchor);
          setTimeout(() => setFlashId(null), 1800);
        } else if (container) {
          container.scrollTo({ top: 0, behavior: "smooth" });
        }
      } else if (scrollRef.current) {
        scrollRef.current.scrollTo({ top: 0, behavior: "smooth" });
      }
    }, 60);
    return () => clearTimeout(t1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jumpSection, jumpStamp]);

  return (
    <div className="paper-wrap">
      <div className="paper-toolbar">
        <div className="paper-tb-left">
          <button className="paper-tb-btn" onClick={() => setPage(p => Math.max(1, p - 1))}>‹</button>
          <span className="paper-pageinfo">
            <span className="m-i"><i>p.</i></span> <em>{page}</em> / 15
          </span>
          <button className="paper-tb-btn" onClick={() => setPage(p => Math.min(15, p + 1))}>›</button>
          <span className="paper-tb-sep"></span>
          <button className="paper-tb-btn" onClick={() => setZoom(z => Math.max(60, z - 10))}>−</button>
          <span className="paper-zoom">{zoom}%</span>
          <button className="paper-tb-btn" onClick={() => setZoom(z => Math.min(150, z + 10))}>+</button>
        </div>
        <div className="paper-tb-right">
          <span className="paper-tb-src">arxiv.org / 1706.03762</span>
          <button className="paper-tb-btn paper-tb-dl">↓ PDF</button>
        </div>
      </div>

      <div className="paper-scroll">
        <div className="paper-page" ref={pageRef} style={{ transform: `scale(${zoom / 100})`, transformOrigin: "top center" }}>
          <div className="paper-page-corner">{page}</div>
          <div className="paper-titleblock">
            <div className="paper-titletop">arXiv:1706.03762v7 [cs.CL] · NeurIPS 2017</div>
            <h2 className="paper-titlemain">Attention Is All You Need</h2>
            <div className="paper-authors-grid">
              <div><div className="au-name">Ashish Vaswani<sup>∗</sup></div><div className="au-aff">Google Brain<br />avaswani@google.com</div></div>
              <div><div className="au-name">Noam Shazeer<sup>∗</sup></div><div className="au-aff">Google Brain<br />noam@google.com</div></div>
              <div><div className="au-name">Niki Parmar<sup>∗</sup></div><div className="au-aff">Google Research<br />nikip@google.com</div></div>
              <div><div className="au-name">Jakob Uszkoreit<sup>∗</sup></div><div className="au-aff">Google Research<br />usz@google.com</div></div>
            </div>
          </div>

          {page === 1 && (
            <>
              <div className="paper-abstract">
                <h3 className="paper-h-center">Abstract</h3>
                <p className="paper-abs-body">
                  This paper introduces a sequence transduction architecture that <span className="paper-mark">dispenses with recurrence and convolutions entirely</span>, relying instead on a self-attention mechanism to draw global dependencies between input and output. Experiments on two WMT 2014 translation tasks demonstrate superior quality alongside materially reduced training time. <span className="paper-elide">[…]</span>
                </p>
              </div>
              <div className="paper-cols">
                <div className="paper-col">
                  <h4 className="paper-h">1&nbsp;&nbsp;Introduction</h4>
                  <p>Recurrent neural networks, long short-term memory, and gated recurrent neural networks have established themselves as state-of-the-art approaches in sequence modeling and transduction problems such as language modeling and machine translation.</p>
                  <p>The inherently sequential nature of recurrent models precludes parallelization within training examples, which becomes critical at longer sequence lengths.</p>
                </div>
                <div className="paper-col">
                  <h4 className="paper-h">2&nbsp;&nbsp;Background</h4>
                  <p>The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet, and ConvS2S, all of which use convolutional neural networks as a basic building block.</p>
                  <p>In these models, the number of operations required to relate signals from two arbitrary positions grows in the distance between them.</p>
                </div>
              </div>
            </>
          )}

          {page === 3 && (
            <div className="paper-cols">
              <div className="paper-col">
                <p className="paper-cont">… of the model. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.</p>
                <h4 className="paper-h" id="sec-3-2">3.2&nbsp;&nbsp;Attention</h4>
                <p>An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.</p>
                <h4 className={"paper-h-sub paper-h-hl" + (flashId === "sec-3-2-1" ? " sec-flash" : "")} id="sec-3-2-1">3.2.1&nbsp;&nbsp;Scaled Dot-Product Attention</h4>
                <p>We call our particular attention <em>Scaled Dot-Product Attention</em> (Figure 2). The input consists of queries and keys of dimension <TeX tex="d_k" />, and values of dimension <TeX tex="d_v" />.</p>
                <p>We compute the matrix of outputs as:</p>
                <div className="paper-eq">
                  <Equation tex="\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V" tag="1" />
                </div>
                <p>The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention.</p>
              </div>
              <div className="paper-col">
                <div className="paper-figure">
                  <div className="paper-fig-art">
                    <div className="fig-row">Scaled Dot-Product Attention</div>
                    <div className="fig-box">MatMul</div>
                    <div className="fig-arrow">↑</div>
                    <div className="fig-box">SoftMax</div>
                    <div className="fig-arrow">↑</div>
                    <div className="fig-box">Mask <span style={{opacity:0.5}}>(opt.)</span></div>
                    <div className="fig-arrow">↑</div>
                    <div className="fig-box">Scale</div>
                    <div className="fig-arrow">↑</div>
                    <div className="fig-box">MatMul</div>
                    <div className="fig-row" style={{display:"flex", gap:18, justifyContent:"center"}}>
                      <span><TeX tex="Q" /></span><span><TeX tex="K" /></span><span><TeX tex="V" /></span>
                    </div>
                  </div>
                  <div className="paper-fig-cap"><em>Figure 2:</em> (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.</div>
                </div>
                <p>While for small values of <TeX tex="d_k" /> the two mechanisms perform similarly, additive attention outperforms dot-product attention without scaling for larger values of <TeX tex="d_k" />.</p>
                <p className="paper-suspect">We suspect that for large values of <TeX tex="d_k" />, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients<sup className="paper-fn paper-fn-link" onClick={() => setFnOpen("4")}>4</sup>. <span className="paper-flag">⚑ flagged by reviewer</span></p>
              </div>
            </div>
          )}

          {page === 4 && (
            <div className="paper-cols">
              <div className="paper-col">
                <h4 className={"paper-h" + (flashId === "sec-3-2-2" ? " sec-flash" : "")} id="sec-3-2-2">3.2.2&nbsp;&nbsp;Multi-Head Attention</h4>
                <p>Instead of performing a single attention function with <TeX tex="d_{\text{model}}" />-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values <em>h</em> times with different, learned linear projections to <TeX tex="d_k" />, <TeX tex="d_k" /> and <TeX tex="d_v" /> dimensions, respectively.</p>
                <p>On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding <TeX tex="d_v" />-dimensional output values.</p>
                <div className="paper-eq">
                  <Equation tex="\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\,W^{O}" tag="2" />
                </div>
                <p>where <TeX tex="\mathrm{head}_i = \mathrm{Attention}(QW_i^{Q},\, KW_i^{K},\, VW_i^{V})" />.</p>
              </div>
              <div className="paper-col">
                <h4 className="paper-h">3.3&nbsp;&nbsp;Position-wise Feed-Forward Networks</h4>
                <p>In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, applied to each position separately and identically.</p>
                <p>This consists of two linear transformations with a ReLU activation in between.</p>
              </div>
            </div>
          )}

          <div className="paper-footer">
            <span>Attention Is All You Need · Vaswani et&nbsp;al. 2017</span>
            <span>· {page} ·</span>
          </div>
        </div>
      </div>

      {fnOpen && PAPER_FOOTNOTES[fnOpen] && (
        <div className="fn-pop" onClick={() => setFnOpen(null)}>
          <div className="fn-pop-card" onClick={(e) => e.stopPropagation()}>
            <div className="fn-pop-head">
              <span className="fn-pop-num">footnote {fnOpen}</span>
              <button className="fn-pop-close" onClick={() => setFnOpen(null)}>×</button>
            </div>
            <div className="fn-pop-body">{PAPER_FOOTNOTES[fnOpen]}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// GRAPH

const GRAPH_NODES = [
  { id: "def-qkv",  kind: "definition", text: "Q, K, V projections",   x: 30,  y: 60,  cls: "def" },
  { id: "asm-iid",  kind: "assumption", text: "i.i.d. unit variance",  x: 250, y: 60,  cls: "def" },
  { id: "step-stab",kind: "step",       text: "softmax stability",     x: 30,  y: 200 },
  { id: "step-var", kind: "step",       text: "variance of QKᵀ",       x: 250, y: 200, tex: "\\text{variance of } QK^{\\top}" },
  { id: "step-con", kind: "step",       text: "concentration bound",   x: 460, y: 200, cls: "suspect" },
  { id: "lem-sqrt", kind: "lemma 1",    text: "√d_k rescale",          x: 140, y: 340, tex: "\\sqrt{d_k}\\ \\text{rescale}" },
  { id: "thm-grad", kind: "theorem 1",  text: "Stable gradient",       x: 140, y: 470, cls: "thm" },
];

const GRAPH_EDGES = [
  ["def-qkv", "step-stab"],
  ["def-qkv", "step-var"],
  ["asm-iid", "step-var"],
  ["asm-iid", "step-con"],
  ["step-stab", "lem-sqrt"],
  ["step-var", "lem-sqrt"],
  ["step-con", "thm-grad", "dashed"],
  ["lem-sqrt", "thm-grad"],
];

function GraphView({ onJumpArticle }) {
  const [focusId, setFocusId] = useState("step-con");
  const [verifyOpen, setVerifyOpen] = useState(false);
  const focused = GRAPH_NODES.find(n => n.id === focusId);

  // compute edge coords from node positions (center-to-center)
  const nodeCenters = {};
  GRAPH_NODES.forEach(n => {
    nodeCenters[n.id] = { x: n.x + 75, y: n.y + 28 };
  });

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <span className="label">filter</span>
        <span className="pill on">all</span>
        <span className="pill">theorem</span>
        <span className="pill">lemma</span>
        <span className="pill">step</span>
        <span className="pill">suspect</span>
        <span style={{ flex: 1 }}></span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>
          7 of 23 nodes · § 3.2 attention
        </span>
      </div>

      <div className="graph-canvas">
        <svg className="edges">
          <defs>
            <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
            </marker>
          </defs>
          {GRAPH_EDGES.map(([a, b, style], i) => {
            const sa = nodeCenters[a], sb = nodeCenters[b];
            const isDash = style === "dashed";
            return (
              <line
                key={i}
                x1={sa.x} y1={sa.y} x2={sb.x} y2={sb.y}
                stroke={isDash ? "var(--accent)" : "var(--ink)"}
                strokeWidth={isDash ? 1.5 : 1.5}
                strokeDasharray={isDash ? "4 4" : null}
                markerEnd="url(#ah)"
                color={isDash ? "var(--accent)" : "var(--ink)"}
              />
            );
          })}
        </svg>

        {GRAPH_NODES.map(n => (
          <div
            key={n.id}
            className={"gnode " + (n.cls || "") + (focusId === n.id ? " focused" : "")}
            style={{ left: n.x, top: n.y }}
            onClick={() => setFocusId(n.id)}
          >
            <span className="kind">{n.kind}</span>
            {n.tex ? <TeX tex={n.tex} /> : <span>{n.text}</span>}
          </div>
        ))}

        {focused && (
          <div className="graph-detail">
            <div className="gd-kind">{focused.kind} · § 3.2.1</div>
            <h4 className="gd-title">{focused.text}</h4>
            {focused.id === "step-con" ? (
              <>
                <p className="gd-text">用 Hoeffding 把 <TeX tex="|QK^{\top}|" /> 的尾部界做掉,得到 softmax 输入的次高斯包络。审查时被标记为 <em style={{ color: "var(--accent)" }}>可疑</em>。</p>
                <div className="gd-quote">
                  "Assume <TeX tex="Q" /> and <TeX tex="K" /> are i.i.d. with zero mean and unit variance; their dot product has zero mean and variance <TeX tex="d_k" />."
                </div>
                <div className="gd-actions">
                  <button className="gd-verify" onClick={() => setVerifyOpen(true)}>⌗ open verification</button>
                  <button className="gd-jump" onClick={() => onJumpArticle && onJumpArticle("§ 5")}>↗ jump to § 5</button>
                </div>
              </>
            ) : (
              <>
                <p className="gd-text">点击其它节点查看依赖与原文引用。每个节点都附 source_quote。</p>
                <button className="gd-jump" onClick={() => onJumpArticle && onJumpArticle("§ 5")}>↗ jump to source</button>
              </>
            )}
          </div>
        )}

        {verifyOpen && (
          <VerificationPanel
            node={focused}
            onClose={() => setVerifyOpen(false)}
            onJumpArticle={onJumpArticle}
          />
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// VERIFICATION PANEL — manual review workspace for a suspect node

function VerificationPanel({ node, onClose, onJumpArticle }) {
  const [verdict, setVerdict] = useState(null); // "ok" | "still" | null
  const [step, setStep] = useState(0); // 0 .. 3
  const STEPS = [
    {
      title: "Re-state the assumption",
      body: <>The bound assumes <TeX tex="Q_i, K_i" /> are <em>independent</em> with mean 0 and variance 1. In practice they are produced by learned projections — independence is at best approximate.</>,
      check: "assume i.i.d. unit-variance"
    },
    {
      title: "Re-derive the variance",
      body: <><TeX tex="\mathrm{Var}(QK^{\top}) = \sum_{i=1}^{d_k} \mathrm{Var}(Q_i K_i) = d_k" display />Stable. The variance step itself is sound.</>,
      check: "variance computation"
    },
    {
      title: "Inspect the tail bound",
      body: <>The paper appeals to Hoeffding-style concentration to claim <TeX tex="QK^{\top}" /> is sub-Gaussian. <strong>This needs boundedness or strict sub-Gaussianity</strong> — for learned projections only an asymptotic statement holds. Confidence cap: <TeX tex="\le 0.7" />.</>,
      check: "concentration bound",
      warn: true
    },
    {
      title: "Try a counter-example",
      body: <>Heavy-tailed inputs (e.g. <TeX tex="Q_i \sim t_{2.5}" />, infinite kurtosis) violate sub-Gaussian; empirically softmax still concentrates because of post-LayerNorm. Bound is <em>practically tight</em>, theoretically <em>looser than claimed</em>.</>,
      check: "counter-example"
    }
  ];
  const cur = STEPS[step];
  return (
    <div className="vrf-overlay" onClick={onClose}>
      <div className="vrf-panel" onClick={(e) => e.stopPropagation()}>
        <div className="vrf-head">
          <div>
            <div className="vrf-eyebrow">VERIFICATION · {node?.kind?.toUpperCase() || "NODE"}</div>
            <h3 className="vrf-title">{node?.text || "node"}</h3>
          </div>
          <button className="vrf-close" onClick={onClose}>×</button>
        </div>

        <div className="vrf-stepper">
          {STEPS.map((s, i) => (
            <button
              key={i}
              className={"vrf-pip" + (i === step ? " cur" : "") + (i < step ? " past" : "")}
              onClick={() => setStep(i)}
            >
              <span className="vrf-pip-n">{String(i + 1).padStart(2, "0")}</span>
              <span className="vrf-pip-l">{s.check}</span>
            </button>
          ))}
        </div>

        <div className={"vrf-step" + (cur.warn ? " warn" : "")}>
          <div className="vrf-step-num">step {step + 1} of {STEPS.length}</div>
          <h4 className="vrf-step-title">{cur.title}</h4>
          <div className="vrf-step-body">{cur.body}</div>
        </div>

        <div className="vrf-foot">
          <button
            className="vrf-nav"
            onClick={() => setStep(s => Math.max(0, s - 1))}
            disabled={step === 0}
          >‹ prev</button>
          {step < STEPS.length - 1 ? (
            <button className="vrf-nav next" onClick={() => setStep(s => s + 1)}>next ›</button>
          ) : (
            <div className="vrf-verdict">
              <button
                className={"vrf-btn ok" + (verdict === "ok" ? " on" : "")}
                onClick={() => setVerdict("ok")}
              >✓ accept</button>
              <button
                className={"vrf-btn still" + (verdict === "still" ? " on" : "")}
                onClick={() => setVerdict("still")}
              >⚑ still suspect</button>
              <button
                className="vrf-jump"
                onClick={() => { onJumpArticle && onJumpArticle("§ 5"); onClose(); }}
              >↗ go to § 5</button>
            </div>
          )}
        </div>

        {verdict && (
          <div className={"vrf-banner " + verdict}>
            {verdict === "ok"
              ? "Marked verified. Confidence raised to 0.95. Logged to vault/notes/1706.03762.md."
              : "Marked still-suspect. Confidence held at 0.7. Added to review queue."}
          </div>
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// DASHBOARD

function DashboardView() {
  // Live counters — elapsed / cost / papers / theme coverage tick over time.
  const [t, setT] = useState(0); // seconds since mount
  useEffect(() => {
    const id = setInterval(() => setT(x => x + 1), 1000);
    return () => clearInterval(id);
  }, []);

  // Base values + slow drift so demo looks alive (loops every ~120s)
  const phase = (t % 120) / 120;            // 0..1
  const elapsedMin = 194 + Math.floor(t / 6); // "3h 14m" baseline = 194min, +1/6s
  const elapsedH = Math.floor(elapsedMin / 60);
  const elapsedM = elapsedMin % 60;
  const elapsedPct = Math.min(99, 54 + phase * 6).toFixed(0);
  const spent = 2.18 + phase * 0.42;
  const spentPct = Math.min(99, 22 + phase * 4).toFixed(0);
  const papers = 14 + Math.floor(phase * 3);
  const papersPct = Math.min(99, 47 + phase * 10).toFixed(0);
  const coverage = phase > 0.6 ? 4 : 3;
  const coveragePct = phase > 0.6 ? 80 : 60;
  const stage3Pct = Math.min(98, 50 + phase * 48).toFixed(0);
  const stage3Spent = (0.54 + phase * 0.18).toFixed(2);
  const stage3Cur = Math.min(4, 2 + Math.floor(phase * 3));

  return (
    <div className="dash-wrap">
      <div className="dash-head">
        <div className="label">RESEARCH SESSION · S_1419</div>
        <h2>"在 Transformer 之后,有哪些路线能把 attention 的 O(n²) 复杂度降到亚二次,而又不显著损失质量?"</h2>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="k">elapsed</div>
          <div className="v">{elapsedH}h <em>{String(elapsedM).padStart(2, "0")}m</em></div>
          <div className="sub">of 6h budget</div>
          <div className="bar" style={{ "--p": elapsedPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">spent</div>
          <div className="v">¥ <em>{spent.toFixed(2)}</em></div>
          <div className="sub">of ¥10.00</div>
          <div className="bar" style={{ "--p": spentPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">papers</div>
          <div className="v"><em>{papers}</em> / 30</div>
          <div className="sub">distilled · graph_depth=step</div>
          <div className="bar" style={{ "--p": papersPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">coverage</div>
          <div className="v"><em>{coverage}</em> / 5</div>
          <div className="sub">themes converged</div>
          <div className="bar" style={{ "--p": coveragePct + "%" }}></div>
        </div>
      </div>

      <div className="pipeline">
        <div className="stage done">
          <span className="stage-num">1 · SEARCH</span>
          <span className="stage-name">42 papers</span>
          <span className="stage-stat">¥0.02</span>
        </div>
        <div className="stage done">
          <span className="stage-num">2 · DISTILL</span>
          <span className="stage-name">14 notes</span>
          <span className="stage-stat">¥1.62</span>
        </div>
        <div className="stage running">
          <span className="stage-num">3 · EXPAND</span>
          <span className="stage-name">{stage3Cur} of 4</span>
          <span className="stage-stat">
            <span className="stage-spin"></span>
            running · ¥{stage3Spent}
          </span>
          <div className="stage-progress" style={{ "--p": stage3Pct + "%" }}></div>
        </div>
        <div className="stage queued">
          <span className="stage-num">4 · CLUSTER</span>
          <span className="stage-name">queued</span>
          <span className="stage-stat">—</span>
        </div>
        <div className="stage queued">
          <span className="stage-num">5 · SYNTH</span>
          <span className="stage-name">queued</span>
          <span className="stage-stat">—</span>
        </div>
      </div>

      <div className="live-log">
        <div className="live-log-head">
          <span>● live</span> expanding paper {stage3Cur}/4 — <em>Linformer</em>
        </div>
        <div className="live-log-line">
          [{String(elapsedH).padStart(2, "0")}:{String(elapsedM).padStart(2, "0")}:{String(t % 60).padStart(2, "0")}] tracing references → found {3 + (t % 5)} new arxiv ids
        </div>
      </div>

      <div className="themes">
        <div className="theme">
          <div className="theme-tag">THEME A</div>
          <h4><em>Sparse</em></h4>
          <p>滑动窗 · 全局+随机 · 块对角</p>
          <div className="papers">Longformer · BigBird · Sparse-T</div>
        </div>
        <div className="theme">
          <div className="theme-tag">THEME B</div>
          <h4><em>Low-rank</em></h4>
          <p>投影到 k 维 · 核函数化</p>
          <div className="papers">Linformer · Performer</div>
        </div>
        <div className="theme">
          <div className="theme-tag">THEME C</div>
          <h4><em>State-space</em></h4>
          <p>替代路线 · 非 attention</p>
          <div className="papers">S4 · Mamba</div>
        </div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// MAIN APP

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [cost, setCost] = useState(0);
  const [tab, setTab] = useState("welcome");
  const [openArticle, setOpenArticle] = useState(false);
  const [hasDashboard, setHasDashboard] = useState(false);
  const [jumpSection, setJumpSection] = useState(null);
  const [jumpStamp, setJumpStamp] = useState(0);
  const [articleFlash, setArticleFlash] = useState(null);
  const [dark, setDark] = useState(() => localStorage.getItem("pd-theme") === "dark");
  const feedRef = useRef(null);

  useEffect(() => {
    document.body.classList.toggle("dark", dark);
    localStorage.setItem("pd-theme", dark ? "dark" : "light");
  }, [dark]);

  // auto-scroll feed
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [messages]);

  const addMsg = (m) => setMessages(prev => [...prev, { ...m, id: m.id || uid() }]);
  const updateMsg = (id, patch) => setMessages(prev => prev.map(m => m.id === id ? { ...m, ...patch } : m));

  const jumpToPaper = (sec) => {
    if (sec) setJumpSection(sec);
    setJumpStamp(s => s + 1);
    setTab("paper");
  };
  const jumpToArticle = (sec) => {
    setOpenArticle(true);
    setTab("article");
    setArticleFlash(sec || null);
    setTimeout(() => {
      const el = sec && document.querySelector(`[data-section="${CSS.escape(sec)}"]`);
      if (el) {
        const container = document.querySelector(".work-body");
        if (container) {
          const top = el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop - 24;
          container.scrollTo({ top, behavior: "smooth" });
        }
      }
      setTimeout(() => setArticleFlash(null), 1600);
    }, 80);
  };

  // Pick a paper from search results — distill it
  const pickPaper = async (paper) => {
    if (busy) return;
    setBusy(true);
    addMsg({ role: "asst", text: `好,开始蒸馏 ${paper.title}。建图深度: step。` });
    await sleep(700);

    const distillId = uid();
    addMsg({ id: distillId, type: "tool", name: "distill_by_id", status: "running",
      args: { ids: `[${paper.arxiv}]`, graph_depth: "step" } });
    setCost(c => c + 0.18);
    await sleep(1400);
    setOpenArticle(true);
    setTab("article");
    await sleep(1800);
    updateMsg(distillId, { status: "done" });
    setCost(c => c + 0.05);
    await sleep(500);
    addMsg({ role: "asst", text: <>蒸馏完成 — 12 段中文笔记 · 23 个证明节点。我注意到 §3.2.1 的 scaled dot-product 推导有个值得展开的地方,已经标好了。<br /><br />要不要我再 review 一遍这篇的证明?</> });
    setBusy(false);
  };

  // Main scripted demo dispatcher
  const sendText = async (text) => {
    if (busy) return;
    addMsg({ role: "user", text });
    setInput("");
    setBusy(true);
    await sleep(500);

    const lower = text.toLowerCase();

    if (text.includes("亚二次") || text.includes("O(n²") || text.includes("O(n^2") || text.includes("深度研究") || text.includes("研究:") || text.includes("研究 ")) {
      const rid = uid();
      addMsg({ id: rid, type: "tool", name: "research", status: "running",
        args: { duration: "6h budget", max_papers: 30, max_cost: "¥10.00" } });
      setCost(c => c + 2.18);
      await sleep(1500);
      addMsg({ role: "asst", text: <>这是个长时任务,我先跑前 3 个阶段:<em>search → distill → expand</em>,然后再聚类综合。当前已蒸馏 14 篇,聚成 3 个主题。可以中途打断 (⌃C) 或调整问题。<br /><br />右边仪表盘开了 →</> });
      setHasDashboard(true);
      setTab("dashboard");
      setBusy(false);
      return;
    }

    if (text.includes("review") || text.includes("审查")) {
      const rid = uid();
      addMsg({ id: rid, type: "tool", name: "review_proof", status: "running",
        args: { target: "1706.03762 · all theorems", conf_cap: "0.7" } });
      setCost(c => c + 0.18);
      setOpenArticle(true);
      await sleep(2400);
      updateMsg(rid, { status: "done" });
      await sleep(300);
      addMsg({ role: "asst", text: <>审查完毕。23 个节点里 <span className="hi">3 个 suspect · 2 个 gap · 1 个被污染</span>。<br /><br />最值得手工核对的是 <span className="hi">concentration bound</span>(§3.2.1)— 它依赖"有界变量"的 Hoeffding 假设,而实际网络中 Q、K 只是近似次高斯。<br /><br />右边图谱已打开,可疑节点带红框。<em>注:审查仅定位,不出具对错判决。</em></> });
      setTab("graph");
      setBusy(false);
      return;
    }

    if (text.includes("√d_k") || text.includes("d_k") || text.includes("scaled")) {
      await sleep(600);
      addMsg({ role: "asst", text: <><TeX tex="\sqrt{d_k}" /> 的作用是稳定 softmax 的梯度。论文 §3.2.1 假设 <TeX tex="Q" />、<TeX tex="K" /> 各分量 i.i.d. 单位方差,则 <TeX tex="QK^{\top}" /> 方差为 <TeX tex="d_k" />。当 <TeX tex="d_k" /> 较大时,未缩放的 logits 会让 softmax 进入饱和区(梯度趋零)。除以 <TeX tex="\sqrt{d_k}" /> 把方差压回 <TeX tex="O(1)" />。<br /><br />我在右边把相关节点和原文引用打开了。</> });
      setOpenArticle(true);
      setTab("article");
      setCost(c => c + 0.003);
      setBusy(false);
      return;
    }

    if (text.includes("Transformer") || text.includes("注意力") || text.includes("蒸馏") || lower.includes("attention")) {
      // Search → present results → wait for pick
      const searchId = uid();
      addMsg({ id: searchId, type: "tool", name: "search", status: "running",
        args: { topic: '"Transformer attention"', n: 5, source: "arxiv (local mirror)" } });
      setCost(c => c + 0.001);
      await sleep(1200);
      updateMsg(searchId, { status: "done", results: SEARCH_RESULTS });
      await sleep(500);
      addMsg({ role: "asst", text: <>找到 5 篇候选 — 上面三篇覆盖<span className="hi">原始架构 · 归一化基础 · 后续改进</span>。点击任一篇开始蒸馏 ↑</> });
      setBusy(false);
      return;
    }

    // Default fallback
    await sleep(800);
    addMsg({ role: "asst", text: "好的,正在思考…(这是一个 demo,试试上方的引导词,或者关于 Transformer attention 的问题。)" });
    setBusy(false);
  };

  const onSend = () => {
    if (input.trim()) sendText(input.trim());
  };
  const onSeed = (text) => sendText(text);

  return (
    <div className="app">
      <TopBar cost={cost} busy={busy} dark={dark} setDark={setDark} />
      <div className="body">
        {/* CHAT */}
        <section className="chat">
          <div className="chat-feed" ref={feedRef}>
            {messages.length === 0 ? (
              <EmptyChat onSeed={onSeed} />
            ) : (
              messages.map(m => {
                if (m.role === "user") return <UserMsg key={m.id} text={m.text} />;
                if (m.role === "asst") return <AsstMsg key={m.id} text={m.text} />;
                if (m.type === "tool") return (
                  <ToolCard
                    key={m.id}
                    name={m.name}
                    status={m.status}
                    args={m.args}
                    results={m.results}
                    onPick={m.name === "search" && m.status === "done" ? pickPaper : null}
                  />
                );
                return null;
              })
            )}
          </div>
          <InputBox value={input} setValue={setInput} onSend={onSend} disabled={busy} />
        </section>

        {/* WORKSPACE */}
        <section className="work">
          <div className="work-tabs">
            <button className={"work-tab" + (tab === "welcome" ? " active" : "")} onClick={() => setTab("welcome")}>
              Welcome
            </button>
            <button
              className={"work-tab" + (tab === "article" ? " active" : "")}
              onClick={() => openArticle && setTab("article")}
              disabled={!openArticle}
            >
              Article{openArticle && <span className="badge">attention</span>}
            </button>
            <button
              className={"work-tab" + (tab === "graph" ? " active" : "")}
              onClick={() => openArticle && setTab("graph")}
              disabled={!openArticle}
            >
              Graph{openArticle && <span className="badge">23 nodes</span>}
            </button>
            <button
              className={"work-tab" + (tab === "paper" ? " active" : "")}
              onClick={() => openArticle && setTab("paper")}
              disabled={!openArticle}
            >
              Paper{openArticle && <span className="badge">pdf · 15p</span>}
            </button>
            <button
              className={"work-tab" + (tab === "dashboard" ? " active" : "")}
              onClick={() => hasDashboard && setTab("dashboard")}
              disabled={!hasDashboard}
            >
              Research{hasDashboard && <span className="badge">s_1419</span>}
            </button>
          </div>
          <div className="work-body">
            {tab === "welcome" && <WelcomeView onPick={() => { setOpenArticle(true); setTab("article"); }} />}
            {tab === "article" && openArticle && (
              <ArticleView
                article={ARTICLE}
                articleFlash={articleFlash}
                onOpenGraph={() => setTab("graph")}
                onOpenPaper={(sec) => jumpToPaper(sec)}
              />
            )}
            {tab === "graph" && openArticle && <GraphView onJumpArticle={jumpToArticle} />}
            {tab === "paper" && openArticle && <PaperView jumpSection={jumpSection} jumpStamp={jumpStamp} />}
            {tab === "dashboard" && hasDashboard && <DashboardView />}
          </div>
        </section>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
