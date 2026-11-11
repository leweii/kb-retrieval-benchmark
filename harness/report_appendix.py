"""Appendix of report.html / report.en.html: methodology, knowledge-base sources, question sources.

Counts (documents per scale, questions per group and scale, clean-set size) are computed from the data files so the
appendix can't drift from what was actually run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCALES = (300, 900, 3000)
LINKS = {
    "musique": "https://huggingface.co/datasets/bdsaglam/musique",
    "musique_src": "https://github.com/StonyBrookNLP/musique",
    "bright": "https://huggingface.co/datasets/xlangai/BRIGHT",
    "nolima": "https://huggingface.co/datasets/amodaresi/NoLiMa",
}


STATS = ROOT / "report/corpus_stats.json"  # document counts per scale, kept so the report builds without corpora/


def counts(memorized: set[str]) -> dict:
    docs = {}
    if (ROOT / "corpora/kb2/S300/docs.jsonl").exists():
        for s in SCALES:
            docs[s] = Counter(json.loads(l)["source"]
                              for l in (ROOT / f"corpora/kb2/S{s}/docs.jsonl").read_text().splitlines())
        STATS.write_text(json.dumps({str(s): dict(c) for s, c in docs.items()}, indent=1))
    else:
        docs = {int(s): Counter(c) for s, c in json.loads(STATS.read_text()).items()}
    qs = [json.loads(l) for l in (ROOT / "questions_v2/questions.jsonl").read_text().splitlines()]
    live = [q for q in qs if not q["excluded"]]
    per = {cat: {s: sum(1 for q in live if q["category"] == cat and s in q.get("valid_scales", q["scales"])) for s in SCALES}
           for cat in ("musique_ans", "musique_unans", "bright_ans", "bright_unans", "nolima")}
    tot = {cat: sum(1 for q in live if q["category"] == cat) for cat in per}
    per_scale = {s: sum(per[c][s] for c in per) for s in SCALES}
    clean = sum(1 for q in live if q["category"] == "musique_ans" and q["qid"] not in memorized)
    audit = json.loads((ROOT / "questions_v2/gold_decisions.json").read_text())
    reasons = [q.get("exclude_reason", "") for q in qs]
    return {"docs": docs, "per": per, "tot": tot, "per_scale": per_scale, "n_all": len(qs), "n_live": len(live),
            "clean": clean, "memorized": len(memorized),
            "gold_drop": sum(v["action"] == "drop" for v in audit.values()),
            "gold_alias": sum(v["action"] == "add_aliases" for v in audit.values()),
            "leak2_mus": sum("check 2" in r or "hand review" in r for r, q in zip(reasons, qs)
                             if q["category"] == "musique_unans"),
            "leak2_bright": sum("check 2" in r for r, q in zip(reasons, qs) if q["category"] == "bright_unans")}


def _rows(rows: list[list[str]], num_cols: set[int] = frozenset()) -> str:
    out = []
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cells = "".join(f'<{tag}{" class=num" if j in num_cols else ""}>{c}</{tag}>' for j, c in enumerate(r))
        out.append(f"<tr>{cells}</tr>")
    return "<table>" + out[0] + "".join(out[1:]) + "</table>"


def robustness_html(lang: str, rob: dict) -> str:
    """Significance table (harness/stats.significance) and recommendation sensitivity to the refusal threshold."""
    zh = lang == "zh"
    pct = lambda x: "–" if x is None else f"{x:.0%}"  # noqa: E731
    head = (["体量", "比较（A vs B）", "题数（簇）", "准确率 A / B", "仅 A 对 / 仅 B 对", "p", "Holm 校正 p", "差值及 95% 区间"]
            if zh else ["Size", "Comparison (A vs B)", "Questions (clusters)", "Accuracy A / B", "Only A / only B right",
                        "p", "Holm-adjusted p", "Difference, 95% interval"])
    rows = [head]
    for r in rob["sig"]:
        sig = r["p_holm"] < 0.05
        rows.append([str(r["scale"]), f"{r['a_label']} vs {r['b_label']}", f"{r['n']} ({r['n_clusters']})",
                     f"{pct(r['acc_a'])} / {pct(r['acc_b'])}", f"{r['b_only']} / {r['c_only']}", f"{r['p']:.3f}",
                     ("<b>" if sig else "") + f"{r['p_holm']:.3f}" + ("</b>" if sig else ""),
                     f"{r['diff'] * 100:+.0f} pt [{r['lo'] * 100:+.0f}, {r['hi'] * 100:+.0f}]"])
    sens = [["体量", "说不知道 ≥ 70%（报告采用）", "≥ 80%"] if zh else ["Size", "Says not in KB ≥ 70% (used)", "≥ 80%"]]
    for scale, picks in rob["sens"].items():
        sens.append([str(scale)] + [f"{x[0]} · {x[1]}（{pct(x[2])}）" if x else "–" for x in picks.values()])
    if zh:
        return f"""<h4>显著性与稳健性</h4>
<p>以下数字由 <code>harness/stats.py</code> 从判分记录计算。配对精确 McNemar 检验只用两格都评过分的同一批干净题；
{len(rob['sig'])} 个比较作为一族做 Holm 校正（粗体 = 校正后 p &lt; 0.05）。共享金标文档或答案的题会互相关联，
差值区间按题目簇做 10,000 次自助重抽样。</p>
{_rows(rows, {2, 5, 6})}
<p>质量首选对"说不知道"门槛的敏感性：</p>
{_rows(sens)}"""
    return f"""<h4>Significance and robustness</h4>
<p>Computed from the judged records by <code>harness/stats.py</code>. Exact paired McNemar tests on the clean questions
graded in both cells; the {len(rob['sig'])} comparisons form one family for the Holm adjustment (bold = adjusted p &lt; 0.05).
Questions sharing a gold document or an answer are not independent, so the interval of the difference resamples
question clusters (10,000 bootstrap draws).</p>
{_rows(rows, {2, 5, 6})}
<p>Quality pick under a stricter correct-refusal threshold:</p>
{_rows(sens)}"""


def appendix_html(lang: str, memorized: set[str], n_judged: int, rob: dict | None = None) -> str:
    c = counts(memorized)
    rb = robustness_html(lang, rob) if rob else ""
    cap = rob["cap"] if rob else {}
    d, per, tot = c["docs"], c["per"], c["tot"]
    sc = lambda cat: " / ".join(str(per[cat][s]) for s in SCALES)  # noqa: E731
    a = lambda k: f'<a href="{LINKS[k]}" target="_blank" rel="noopener">{LINKS[k].split("//")[1]}</a>'  # noqa: E731
    if lang == "zh":
        return f"""<section class="appx">
<h2>附录</h2>
<h3>A · 实验方法</h3>
<h4>对比的维度</h4>
{_rows([["维度", "取值"],
        ["知识库体量", "300 / 900 / 3000 篇，嵌套（300 ⊂ 900 ⊂ 3000），文档组成见附录 B"],
        ["模型", "Claude Haiku 5.5、Claude Sonnet 5.5（Anthropic API，effort medium）；DeepSeek V4.1 Flash（DeepSeek 官方 API，默认 effort high，该 API 没有 medium 档）"],
        ["检索方式", "R1 grep（正则搜全文 + 读文件）· R2 BM25 · R3 混合（BM25 + 向量，RRF 融合）· R4 增强混合（QMD 式：Haiku 查询扩展 → BM25 + 向量 → RRF → Haiku 重排 30 个候选）· R5 LLM-wiki 导航（从 index.md 顺链接读，可回读原文）· R6 LLM-wiki + 混合检索"],
        ["使用模式", "单次：用问题检索一次，取前 8 段交给模型作答 · 两轮：材料不够时，模型联想出 1–3 个新查询再检索一次，仍没有就拒答 · 智能体：模型自己调用工具（搜索 / grep / 列目录 / 读文件），最多 12 步"]])}
<h4>组件</h4>
<ul>
<li>向量：<code>qwen3-embedding-8b</code>（OpenRouter，固定服务商 DeepInfra）；本地 numpy 暴力检索，不用向量数据库。切块按 markdown 标题，约 600 token。</li>
<li>LLM-wiki：Claude Haiku 5.5 逐篇提炼（摘要、关键事实、实体），代码汇总成 index / 主题页 / 摘要页 / 实体页。50 篇盲评中 Haiku 与 Sonnet 提炼质量相当。</li>
<li>统一提示词：具体事实必须来自知识库；可以用常识理解和串联知识库内容，不能用常识补缺；库里没有要明确说。智能体检索策略：先搜关键词（非英文问题同时用英文关键词）→ 没结果就联想后再搜 → 多步问题逐块找 → 仍没有才拒答。</li>
</ul>
<h4>评分与指标</h4>
{_rows([["指标", "定义"],
        ["准确率", f"MuSiQue 可回答题的答案是否与标准答案或别名一致（Claude Opus 5.5 判定）。只统计 {c['clean']} 道干净题：剔除了任一被测模型不查知识库就能答对的 {c['memorized']} 道"],
        ["找全证据", "多跳题的全部证据文档都曾展示给模型：检索段落、搜索摘要、grep 命中行或打开读取（wiki 摘要页映射回原文）。这个比例随搜索 / grep 次数机械增长（一次 grep 最多列出 60 个文件），只说明证据出现过，不代表搜得更彻底"],
        ["误拒", "MuSiQue / BRIGHT 可回答题，回答\"知识库里没有\"的比例"],
        ["说\"不知道\"", "不可回答题中明确表示知识库没有的比例；直接给出答案算幻觉"],
        ["p50 耗时", "从提问到最终答案的端到端时间中位数，多进程并发负载下测得"],
        ["单题花费", "模型调用 + 检索 API（embedding、R4 的查询扩展和重排）按各家实际价格计算"],
        ["轮次", "每道题调用被测模型的次数：单次 = 1；两轮 = 1–3；智能体每轮一次调用，最多 12 轮，用满后再强制作答 1 次（因此最多 13）。\"用满步数\"= 达到 12 轮上限的比例"],
        ["工具调用次数", "每道题调用搜索 / grep / 列目录 / 读文件等工具的总次数（两轮模式的\"再搜一次\"算 1 次）；智能体一轮可以并行调用多个工具"]])}
<ul>
<li>评分：Claude Opus 5.5（不在被测模型之列），Message Batches；共 {n_judged:,} 条回答全部评分。BRIGHT 是开放问题、NoLiMa 主要考联想检索，这两组只评检索和拒答，不评答案准确率。没有给出回答的运行（报错或空答案）按失败计：可回答题算答错、算未作答，不可回答题不算正确拒答。</li>
<li>推荐规则：只在\"说不知道 ≥ 70% 且已评分 MuSiQue ≥ 30 题\"的组合中比较；质量首选 = 准确率最高；性价比 = 与首选差距 ≤ 5 个百分点中最便宜；Flash 级 = 只在 Haiku / DeepSeek 中选。</li>
<li>关键差异的检验见下方"显著性与稳健性"。</li>
</ul>
{rb}
<h4>局限</h4>
<ul>
<li>公开数据集不是个人 md 笔记：同一项目多篇笔记、版本冲突、写法与问法不一致等个人知识库特有的难点没有覆盖。</li>
<li>Claude 用 effort medium、DeepSeek 用 high，思考强度不完全对等。</li>
<li>闭卷记忆过滤：出题时只用 Sonnet 过滤；Haiku 和 DeepSeek 是事后补测的，DeepSeek 有 24 条闭卷预测为空，相当于没测。</li>
<li>体量之间不只差干扰文档的数量（见附录 B）：300 / 900 篇里几乎没有同题的易混淆文档，体量曲线同时混入了这个因素。</li>
<li>中文题保留了英文专有名词（多数含金标文档标题），基本没有测到跨语言检索。</li>
<li>耗时在并发负载下测得，不同 API 通道之间不宜直接比较。智能体最多 12 步；用满步数的比例：{"、".join(f"{k} {v:.0%}" for k, v in cap.items())}，结果对步数上限敏感。</li>
<li>R4 的重排用 Haiku 代替 QMD 原版的 Qwen3-Reranker（16 题对比召回相同）。千问 3.8 Flash 因 API 额度未参与。</li>
</ul>
<h3>B · 知识库来源</h3>
{_rows([["数据集", "在知识库里是什么", "300 篇", "900 篇", "3000 篇", "许可", "来源"],
        ["MuSiQue（dev，full 版）", "维基百科段落，同一词条的段落合并为一篇", str(d[300]["wiki"]), str(d[900]["wiki"]), str(d[3000]["wiki"]), "CC BY 4.0", a("musique") + "<br>" + a("musique_src")],
        ["BRIGHT（biology、earth_science、economics、psychology、robotics、stackoverflow、sustainable_living）", "StackExchange 帖子引用的网页原文（long_documents），中位约 1.3 万字符，含导航等噪声", str(d[300]["bright"]), str(d[900]["bright"]), str(d[3000]["bright"]), "CC BY 4.0", a("bright")],
        ["NoLiMa（III-filter 书本 + needle set）", "小说切成约 500 词一篇，部分片段插入一句\"植入句\"", str(d[300]["books"]), str(d[900]["books"]), str(d[3000]["books"]), "Adobe Research License（非商业研究）", a("nolima")]],
       {2, 3, 4})}
<ul>
<li>每篇存成一个 .md 文件（frontmatter + 标题），按 来源 / 领域 分文件夹。体量不只是"干扰文档变多"：金标文档占全库 72%（300 篇）、65%（900 篇）、20%（3000 篇）；300 篇的维基部分 165 篇里 162 篇是金标，同题的易混淆维基段落基本只在 3000 篇里出现。</li>
<li>不可回答题对应的 84 篇证据文档（\"缺失的一环\"）不进入任何一档知识库。</li>
<li>报告只包含汇总指标，不包含数据集原文。</li>
</ul>
<h3>C · 题目来源</h3>
{_rows([["题组", "有效题数", "每档提问数（300 / 900 / 3000）", "来源与构造"],
        ["MuSiQue 可回答", str(tot["musique_ans"]), sc("musique_ans"), f"MuSiQue dev 的 2–4 跳问题；Sonnet 闭卷能答对的题不选。金标经 Claude Opus 5.5 盲审（只看题目、金标、分解链和金标文档，不看任何模型的回答，以知识库为准）：删去 {c['gold_drop']} 道问题不成立或金标错误的题，为 {c['gold_alias']} 道补充库里同样支持的答案作别名"],
        ["MuSiQue 不可回答", str(tot["musique_unans"]), sc("musique_unans"), f"取可回答题，把最后一跳的证据页从所有知识库中删除（\"缺一环\"）。泄漏检查两轮：Sonnet 按整题检索候选文档判断；Opus 再按被删的最后一跳子问题检索、全库搜原答案，并人工复核大多数运行都作答的题——第二轮又排除 {c['leak2_mus']} 道（按体量）"],
        ["BRIGHT 可回答", str(tot["bright_ans"]), sc("bright_ans"), "StackExchange 真实提问，标准证据为其引用的网页"],
        ["BRIGHT 不可回答", str(tot["bright_unans"]), sc("bright_unans"), f"证据网页移出知识库；按体量逐档检查剩余文档能否给出实质性回答，能回答的那一档不计入。第一轮只看了每篇文档前 1.2 万字符，第二轮按切块不截断、并加入作答运行读过的文档，又在部分体量排除 {c['leak2_bright']} 道"],
        ["NoLiMa", str(tot["nolima"]), sc("nolima"), "核心题 4 个植入句模板，每个出单跳和双跳两题；900 / 3000 篇另加扩展植入句，与核心题同模板，因此 20 道中 15 道的问句在库里有多个合法答案（如 \"Which character has been to France?\" 对应 3 个角色），只按其中一个评分，NoLiMa 检索指标在 900 / 3000 篇偏低"]],
       {1, 2})}
<ul>
<li>共 {c['n_all']} 道，有效 {c['n_live']} 道（泄漏检查和金标审核排除 {c['n_all'] - c['n_live']} 道，部分题只在个别体量排除）；每档实际提问：300 篇 {c['per_scale'][300]} 道（核心题），900 篇 {c['per_scale'][900]} 道、3000 篇 {c['per_scale'][3000]} 道（核心 + 扩展）。</li>
<li>约一半题目由 Claude Sonnet 5.5 翻译成中文（人名、机构名等专有名词保留英文），用来模拟\"中文提问、英文知识库\"。</li>
<li>选数据集前做过区分度检查：RepliQA、MultiHop-RAG 的题目是看着文档写的（题干实词 80% 以上原样出现在答案文档中，BM25 召回 100%），HotpotQA、SciFact、DuRetrieval、T2Retrieval 的 BM25 召回都在 90% 以上，均未采用。</li>
</ul>
</section>"""
    return f"""<section class="appx">
<h2>Appendix</h2>
<h3>A · Method</h3>
<h4>What was compared</h4>
{_rows([["Dimension", "Values"],
        ["Knowledge-base size", "300 / 900 / 3000 documents, nested (300 ⊂ 900 ⊂ 3000); composition in Appendix B"],
        ["Models", "Claude Haiku 5.5, Claude Sonnet 5.5 (Anthropic API, effort medium); DeepSeek V4.1 Flash (DeepSeek API, default effort high — that API has no medium level)"],
        ["Retrieval", "R1 grep (regex over files + read) · R2 BM25 · R3 hybrid (BM25 + dense, RRF) · R4 enhanced hybrid (QMD-style: Haiku query expansion → BM25 + dense → RRF → Haiku rerank of 30 candidates) · R5 LLM-wiki navigation (start at index.md, follow links, may open raw notes) · R6 LLM-wiki + hybrid search"],
        ["Mode", "Single-shot: retrieve once with the question, top-8 passages to the model · Two-round: if insufficient, the model writes 1–3 associative queries for one more retrieval, then answers or refuses · Agent: the model calls tools (search / grep / list / read) for up to 12 steps"]])}
<h4>Components</h4>
<ul>
<li>Embeddings: <code>qwen3-embedding-8b</code> (OpenRouter, pinned to DeepInfra); brute-force cosine in numpy, no vector database. Chunks split on markdown headings, ~600 tokens.</li>
<li>LLM-wiki: Claude Haiku 5.5 digests each note (summary, key facts, entities); code assembles index / topic / source / entity pages. In a 50-document blind review Haiku and Sonnet digests were on par.</li>
<li>Shared prompt: facts must come from the knowledge base; general knowledge may interpret and connect KB content but not fill gaps; say clearly when the KB lacks the answer. Agent search strategy: key terms (plus English keywords for non-English questions) → associations if nothing relevant → each missing piece of a multi-step question → only then refuse.</li>
</ul>
<h4>Grading and metrics</h4>
{_rows([["Metric", "Definition"],
        ["Accuracy", f"MuSiQue answerable questions: answer matches the reference or an alias (judged by Claude Opus 5.5). Only the {c['clean']} clean questions count: {c['memorized']} questions that any tested model answers without the KB were removed"],
        ["Full evidence found", "Every evidence document of a multi-hop question was shown to the model: retrieved passage, search snippet, grep hit line or opened file (wiki source pages map back to their raw note). It grows mechanically with the number of searches / grep calls (one grep call can list 60 files): it shows the evidence came up, not that the search was more thorough"],
        ["False refusal", "Share of MuSiQue / BRIGHT answerable questions answered with \"not in the KB\""],
        ["Says \"not in the KB\"", "Share of unanswerable questions where the answer clearly states the KB lacks it; answering anyway counts as hallucination"],
        ["p50 latency", "Median end-to-end time from question to final answer, measured under concurrent load"],
        ["Cost per question", "Model calls + retrieval APIs (embeddings, R4 expansion and reranking) at each provider's actual prices"],
        ["Rounds", "Calls to the tested model per question: single-shot = 1; two-round = 1–3; agent = one call per round, up to 12 rounds plus one forced answer at the cap (max 13). \"Hit step cap\" = share reaching the 12-round cap"],
        ["Tool calls", "Total search / grep / list / read calls per question (the two-round extra search counts as 1); an agent may call several tools per round"]])}
<ul>
<li>Grader: Claude Opus 5.5 (not a tested model), via Message Batches; all {n_judged:,} answers graded. BRIGHT is open-ended and NoLiMa mainly tests associative retrieval, so these two are graded on retrieval and refusal only. Runs without a usable response (error or empty answer) count as failures: wrong and not answered on answerable questions, not a correct refusal on unanswerable ones.</li>
<li>Recommendation rule: among combinations with ≥ 70% correct refusals and ≥ 30 graded MuSiQue questions — best quality = highest accuracy; best value = cheapest within 5 points of it; Flash-tier = Haiku / DeepSeek only.</li>
<li>Tests of the key differences: see "Significance and robustness" below.</li>
</ul>
{rb}
<h4>Limitations</h4>
<ul>
<li>Public datasets are not personal md notes: many notes on one project, conflicting versions and wording mismatch between writing and asking are not covered.</li>
<li>Claude ran at effort medium, DeepSeek at high — reasoning budgets are not fully matched.</li>
<li>Closed-book memory filter: questions were selected with Sonnet only; Haiku and DeepSeek were checked afterwards, and 24 of DeepSeek's closed-book predictions were empty, i.e. not tested.</li>
<li>Sizes differ in more than the number of distractors (Appendix B): 300 / 900 docs contain almost none of the questions' own confusable pages, so the size curve mixes in that factor.</li>
<li>Chinese questions keep English proper nouns (most contain a gold document's title), so cross-language retrieval is barely tested.</li>
<li>Latency was measured under concurrent load; compare across API providers with care. Agents were capped at 12 steps; share of agent runs that hit the cap: {", ".join(f"{k} {v:.0%}" for k, v in cap.items())} — results are sensitive to the cap.</li>
<li>R4 reranks with Haiku instead of QMD's Qwen3-Reranker (same recall in a 16-query check). Qwen 3.8 Flash was not run (API credits).</li>
</ul>
<h3>B · Knowledge-base sources</h3>
{_rows([["Dataset", "What a document is", "300", "900", "3000", "License", "Source"],
        ["MuSiQue (dev, full)", "Wikipedia paragraphs, merged per article title", str(d[300]["wiki"]), str(d[900]["wiki"]), str(d[3000]["wiki"]), "CC BY 4.0", a("musique") + "<br>" + a("musique_src")],
        ["BRIGHT (biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living)", "Web pages cited by StackExchange posts (long_documents), median ~13k characters incl. navigation noise", str(d[300]["bright"]), str(d[900]["bright"]), str(d[3000]["bright"]), "CC BY 4.0", a("bright")],
        ["NoLiMa (III-filter books + needle set)", "~500-word fiction chunks; some carry one inserted needle sentence", str(d[300]["books"]), str(d[900]["books"]), str(d[3000]["books"]), "Adobe Research License (non-commercial research)", a("nolima")]],
       {2, 3, 4})}
<ul>
<li>One .md file per document (frontmatter + title), folders by source / domain. Size is not only \"more distractors\": gold documents make up 72% (300), 65% (900) and 20% (3,000) of the KB; 162 of the 165 Wikipedia pages at 300 docs are gold, and the questions' own confusable Wikipedia paragraphs appear almost only at 3,000.</li>
<li>The 84 evidence documents behind unanswerable questions (the \"missing link\") are in no knowledge base.</li>
<li>The report contains aggregate metrics only, no dataset text.</li>
</ul>
<h3>C · Question sources</h3>
{_rows([["Group", "Questions", "Asked at 300 / 900 / 3000", "Source and construction"],
        ["MuSiQue answerable", str(tot["musique_ans"]), sc("musique_ans"), f"MuSiQue dev 2–4-hop questions; those Sonnet answers closed-book were not selected. Gold labels audited blind by Claude Opus 5.5 (question, reference, decomposition and gold documents only — no model answers; judged against the KB): {c['gold_drop']} ill-posed or mislabeled questions removed, {c['gold_alias']} given aliases for other answers the KB supports"],
        ["MuSiQue unanswerable", str(tot["musique_unans"]), sc("musique_unans"), f"An answerable question with its last-hop evidence page removed from every KB. Two leak checks: Sonnet judged candidates retrieved with the full question; Opus then retrieved with the removed last-hop sub-question, searched the whole KB for the original answer, and questions most runs answered were reviewed by hand — {c['leak2_mus']} more removed (per size)"],
        ["BRIGHT answerable", str(tot["bright_ans"]), sc("bright_ans"), "Real StackExchange questions; gold evidence = the web pages they cite"],
        ["BRIGHT unanswerable", str(tot["bright_unans"]), sc("bright_unans"), f"Evidence pages removed; checked per size whether remaining documents allow a substantive answer — not counted at sizes where they do. The first check saw only each page's first 12k characters; a second pass on untruncated chunks, plus the documents answering runs read, removed {c['leak2_bright']} more at some sizes"],
        ["NoLiMa", str(tot["nolima"]), sc("nolima"), "Core: 4 needle templates, a one-hop and a two-hop question each. The extended needles added at 900 / 3,000 docs reuse the core templates, so 15 of the 20 questions have several valid answers in the KB (\"Which character has been to France?\" fits 3 characters) but are graded against one; NoLiMa retrieval numbers at 900 / 3,000 are biased low"]],
       {1, 2})}
<ul>
<li>{c['n_all']} questions, {c['n_live']} valid ({c['n_all'] - c['n_live']} excluded by the leak checks and the gold audit; some only at some sizes). Asked per size: {c['per_scale'][300]} at 300 docs (core), {c['per_scale'][900]} at 900 and {c['per_scale'][3000]} at 3,000 (core + extended).</li>
<li>About half were translated to Chinese by Claude Sonnet 5.5 (proper nouns kept in English) to model asking an English KB in Chinese.</li>
<li>Datasets were screened for discriminative power first: RepliQA and MultiHop-RAG questions were written from the documents (80%+ of question content words appear verbatim in the gold document; BM25 recall 100%); HotpotQA, SciFact, DuRetrieval and T2Retrieval had BM25 recall above 90%. None were used.</li>
</ul>
</section>"""
