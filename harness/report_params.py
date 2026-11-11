"""Report appendix section: run parameters per model (thinking effort, caching, limits, prices) and measured prompt-cache
hit rates per model x mode.

Cache statistics come from the graded run records (results/p2_*.judged.jsonl, per-question token counts) and are
cached in report/run_params.json so the report builds without the records. Hit rate = cache-read tokens / all prompt
tokens (uncached input + cache read + cache write). "Prompt cost vs no cache" prices the same tokens with and without
caching (Claude base-tier prices; DeepSeek peak prices -- off-peak halves every price, so the ratio is the same).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from llm import CLAUDE_PRICES, DEEPSEEK_PRICES, EFFORT, MODELS  # noqa: E402

CACHE = ROOT / "report/run_params.json"
MODES = ["A0", "A2", "A1"]
ORDER = ["haiku", "sonnet", "deepseek"]


def cache_stats() -> dict:
    files = sorted(glob.glob(str(ROOT / "results/p2_*.judged.jsonl")))
    if not files:
        return json.loads(CACHE.read_text())
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # uncached input, cache read, cache write, output, n
    seen = set()
    for f in files:
        for line in open(f):
            r = json.loads(line)
            t = r.get("tokens")
            k = (r["model"], r["qid"], r["mode"], r["config"], r["scale"])
            if not t or k in seen:
                continue
            seen.add(k)
            a = agg[(r["model"], r["mode"])]
            a[0] += t["input"]; a[1] += t["cache_read"]; a[2] += t["cache_write"]; a[3] += t["output"]; a[4] += 1
    out = {}
    for (m, mode), (i, cr, cw, o, n) in agg.items():
        tot = i + cr + cw
        if MODELS[m]["provider"] == "anthropic":
            pin, _, pcr, pcw = CLAUDE_PRICES[MODELS[m]["id"]]
            actual, nocache = i * pin + cr * pcr + cw * pcw, tot * pin
        else:
            ph, pm, _ = DEEPSEEK_PRICES["peak"]
            actual, nocache = i * pm + cr * ph, tot * pm
        out[f"{m}|{mode}"] = {"n": n, "prompt_per_q": tot / n, "output_per_q": o / n, "hit": cr / tot,
                              "write": cw / tot, "cost_ratio": actual / nocache}
    CACHE.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def _table(head: list[str], rows: list[list[str]], num: set[int]) -> str:
    th = "".join(f'<th class="num">{h}</th>' if j in num else f"<th>{h}</th>" for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if j in num else f"<td>{c}</td>"
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _price(m: str) -> str:
    if MODELS[m]["provider"] == "anthropic":
        pin, pout, pcr, pcw = CLAUDE_PRICES[MODELS[m]["id"]]
        return f"${pin:g} / ${pout:g} · ${pcr:g} / ${pcw:g}"
    ph, pm, po = DEEPSEEK_PRICES["peak"]
    return f"${pm:g} / ${po:g} · ${ph:g} / —"


TXT = {
    "zh": {
        "h": "D · 测试参数",
        "intro": "影响结果的关键运行参数。三个模型的思考强度不一致（Claude medium、DeepSeek 默认 high），"
                 "跨模型比较时要考虑这一点。",
        "head": ["", "Claude Haiku 5.5", "Claude Sonnet 5.5", "DeepSeek V4.1 Flash"],
        "rows": lambda: [
            ["API / 模型 ID", "Anthropic · <code>claude-haiku-5-5</code>", "Anthropic · <code>claude-sonnet-5-5</code>",
             "DeepSeek 官方（OpenAI 兼容）· <code>deepseek-flash</code>"],
            ["思考强度", f"<code>effort: {EFFORT}</code>", f"<code>effort: {EFFORT}</code>",
             "未传参数，用 API 默认 <b>high</b>（只支持 low / high / max，没有 medium）"],
            ["思考是否跨轮保留", "保留（thinking 块原样带回下一轮）", "保留（同左）",
             "不保留（每轮只带回答案文本和工具调用，<code>reasoning_content</code> 不回传）"],
            ["提示缓存", "顶层 <code>cache_control: ephemeral</code>（5 分钟），自动在最后一个可缓存块打断点",
             "同左", "API 自动缓存（无需参数）"],
            ["价格 $/百万 token<br><span class=\"muted\">输入 / 输出 · 缓存读 / 缓存写</span>",
             _price("haiku") + "<br><span class=\"muted\">提示 &gt; 100K 时 ×5</span>", _price("sonnet"),
             _price("deepseek") + "<br><span class=\"muted\">高峰价：北京时间工作日 9–12 点、14–18 点；其余时段半价</span>"],
            ["其余设置", "<code>max_tokens</code> 16000，temperature 默认", "同左", "同左"],
        ],
        "common": [
            "单次：取检索结果前 8 段放进提示，调用 1 次模型。两轮：前 8 段 + 最多 1 次补充检索（1–3 个新查询）。"
            "智能体：最多 12 轮模型调用，用满后再强制作答 1 次。",
            "增强混合（R4）：BM25 + 向量各取 40 条，Haiku 生成扩展查询（<code>effort: low</code>），RRF 融合后取前 30 条交给 "
            "Haiku 重排（<code>effort: low</code>）。向量模型 <code>qwen/qwen3-embedding-8b</code>（OpenRouter，固定 DeepInfra 通道）。",
            "评分：Claude Opus 5.5，<code>effort: low</code>，JSON schema 输出，Message Batches。",
            "耗时是多线程并发跑的时候测的（默认 8 个线程），只能大致对比。",
        ],
        "cache_h": "提示缓存命中",
        "cache_intro": "命中率 = 缓存读 token ÷ 全部提示 token。‘相对不缓存的提示费用’：同样的 token 按不开缓存计价，实际花了多少比例。统计全部运行过的回答（包括审计后剔除的题），所以回答数比页首的评分数多。",
        "cache_head": ["模型", "模式", "回答数", "提示 token / 题", "缓存命中", "缓存写入", "输出 token / 题", "相对不缓存的提示费用"],
        "modes": {"A0": "单次", "A2": "两轮", "A1": "智能体"},
        "cache_notes": [
            "智能体模式下每轮都要重发前面的对话，缓存命中最高：DeepSeek 82%，Claude 64–67%。"
            "DeepSeek 缓存读只要正常输入价格的 1/50，所以它虽然平均每题处理约 25 万提示 token，单题花费仍然最低。",
            "Claude 单次模式只调用一次，写进缓存的提示永远读不到，又要付 1.25 倍的写入价，这部分提示费用多花了约 25%。"
            "这是测试设置的问题，实际部署单次检索时应关掉缓存。",
        ],
    },
    "en": {
        "h": "D · Run parameters",
        "intro": "Key run parameters that affect the results. Thinking effort differs between models (Claude medium, "
                 "DeepSeek default high); keep this in mind when comparing models.",
        "head": ["", "Claude Haiku 5.5", "Claude Sonnet 5.5", "DeepSeek V4.1 Flash"],
        "rows": lambda: [
            ["API / model ID", "Anthropic · <code>claude-haiku-5-5</code>", "Anthropic · <code>claude-sonnet-5-5</code>",
             "DeepSeek official (OpenAI-compatible) · <code>deepseek-flash</code>"],
            ["Thinking effort", f"<code>effort: {EFFORT}</code>", f"<code>effort: {EFFORT}</code>",
             "not set: API default <b>high</b> (only low / high / max exist, no medium)"],
            ["Thinking kept across turns", "yes (thinking blocks sent back unchanged)", "yes (same)",
             "no (each turn sends back only the answer text and tool calls; <code>reasoning_content</code> is dropped)"],
            ["Prompt caching", "top-level <code>cache_control: ephemeral</code> (5 min), breakpoint on the last cacheable block",
             "same", "automatic on the API (no parameter)"],
            ["Price, $ per M tokens<br><span class=\"muted\">input / output · cache read / cache write</span>",
             _price("haiku") + "<br><span class=\"muted\">×5 for prompts over 100K</span>", _price("sonnet"),
             _price("deepseek") + "<br><span class=\"muted\">peak price weekdays 01–04 and 06–10 UTC; half price otherwise</span>"],
            ["Other", "<code>max_tokens</code> 16000, default temperature", "same", "same"],
        ],
        "common": [
            "Single-shot: top-8 retrieved passages in the prompt, one model call. Two-round: top-8 passages + at most one "
            "extra search (1–3 new queries). Agent: at most 12 model calls, then one forced answer.",
            "Enhanced hybrid (R4): 40 candidates each from BM25 and dense, Haiku query expansion (<code>effort: low</code>), "
            "RRF fusion, top 30 reranked by Haiku (<code>effort: low</code>). Embeddings: <code>qwen/qwen3-embedding-8b</code> "
            "(OpenRouter, pinned to DeepInfra).",
            "Grader: Claude Opus 5.5, <code>effort: low</code>, JSON-schema output, Message Batches.",
            "Latency was measured with concurrent workers (8 by default), so it is only roughly comparable.",
        ],
        "cache_h": "Prompt cache hits",
        "cache_intro": "Hit rate = cache-read tokens ÷ all prompt tokens. 'Prompt cost vs no cache': what was actually paid "
                       "for prompt tokens as a share of the price without caching. Counts every answer that was run, including questions "
                       "later removed by the audit, so the totals exceed the graded count at the top.",
        "cache_head": ["Model", "Mode", "Answers", "Prompt tokens / q", "Cache hit", "Cache write", "Output tokens / q",
                       "Prompt cost vs no cache"],
        "modes": {"A0": "Single-shot", "A2": "Two-round", "A1": "Agent"},
        "cache_notes": [
            "Agents resend the conversation every round, so they hit the cache most: DeepSeek 82%, Claude 64–67%. "
            "DeepSeek cache reads cost 1/50 of normal input, which is why it stays cheapest per question while processing "
            "about 250K prompt tokens per question on average.",
            "Claude single-shot makes one call, so the cached prompt is never read but still pays the 1.25× write price: "
            "about 25% extra on prompt cost. This is an artefact of the test setup; turn caching off for single-shot "
            "retrieval in production.",
        ],
    },
}

LABEL = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "deepseek": "DeepSeek V4.1 Flash"}


def _li(xs: list[str], cls: str = "") -> str:
    return f'<ul{f" class={cls!r}" if cls else ""}>' + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"


def params_html(lang: str) -> str:
    """Appendix D: run parameters per model."""
    t = TXT[lang]
    return f'<h3>{t["h"]}</h3><p>{t["intro"]}</p>' + _table(t["head"], t["rows"](), set()) + _li(t["common"])


def cache_html(lang: str) -> str:
    """Main-body section (below the data table): measured prompt-cache hits per model x mode."""
    t, st = TXT[lang], cache_stats()
    rows = []
    for m in ORDER:
        for mode in MODES:
            s = st.get(f"{m}|{mode}")
            if not s:
                continue
            rows.append([LABEL[m], t["modes"][mode], f"{s['n']:,}", f"{s['prompt_per_q']:,.0f}", f"{s['hit']:.0%}",
                         f"{s['write']:.0%}", f"{s['output_per_q']:,.0f}", f"{s['cost_ratio']:.0%}"])
    return (f'<h2>{t["cache_h"]}</h2><div class="sub">{t["cache_intro"]}</div>'
            f'<div id="cache">{_table(t["cache_head"], rows, {2, 3, 4, 5, 6, 7})}</div>' + _li(t["cache_notes"], "notes"))


if __name__ == "__main__":
    print(json.dumps(cache_stats(), indent=1))
