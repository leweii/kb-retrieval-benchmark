"""report.html / report.en.html — core-metric comparison (no external libraries; data embedded).

Metrics per model x configuration x scale: MuSiQue accuracy, full-evidence recall, false refusal, correct refusal on
unanswerable questions, p50 latency, cost per question. Reuses build_report.Data / recommend().
  uv run python -I harness/build_html.py   -> report.html (Chinese) and report.en.html (English)
All page text comes from STRINGS[lang]: {{key}} placeholders in the HTML, the T object in the script.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT))
from build_report import CONFIGS, MODEL_LABEL, MODELS, SCALES, TOTAL_PER_SCALE, Data, recommend  # noqa: E402
from report_appendix import appendix_html  # noqa: E402

MEMORIZED: set[str] = set()  # filled by collect(); used by the appendix


CONFIG_EN = {
    "单次 · BM25": "Single-shot · BM25", "单次 · 混合": "Single-shot · Hybrid",
    "单次 · 增强混合(QMD式)": "Single-shot · Enhanced hybrid (QMD)",
    "两轮 · BM25": "Two-round · BM25", "两轮 · 混合": "Two-round · Hybrid",
    "智能体 · grep": "Agent · grep", "智能体 · BM25": "Agent · BM25", "智能体 · 混合": "Agent · Hybrid",
    "智能体 · 增强混合": "Agent · Enhanced hybrid", "智能体 · LLM-wiki 导航": "Agent · LLM-wiki navigation",
    "智能体 · LLM-wiki+混合": "Agent · LLM-wiki + hybrid",
}

STRINGS = {
    "zh": {
        "lang": "zh", "other_href": "report.en.html", "other_lang": "en", "other_label": "English", "title": "知识库方案评测", "h1": "知识库方案评测 · 核心指标", "recs_h": "推荐", "core_h": "核心指标",
        "acc_t": "MuSiQue 准确率", "acc_s": "越高越好 · 多跳问答，短答案 · 已剔除任一模型不查库就能答对的题",
        "all_t": "找全证据", "all_s": "越高越好 · 多跳题的所有证据都被检索到或读到",
        "fr_t": "误拒", "fr_s": "越低越好 · 库里有答案却说\"没有\"",
        "ro_t": "说\"不知道\"", "ro_s": "越高越好 · 库里没有答案时正确拒答",
        "value_h": "性价比", "cost_t": "准确率 vs 单题花费", "cost_s": "横轴对数刻度 · 左上角最好",
        "lat_t": "准确率 vs 耗时（p50）", "lat_s": "左上角最好 · 并发运行下测得，跨模型通道仅供参考",
        "table_h": "数据表", "k_w": 88,
        "rounds_h": "轮询次数", "steps_t": "平均轮次（模型调用次数）",
        "steps_s": "越少越快越省 · 单次 = 1；两轮 ≤ 3；智能体 ≤ 12 轮（用满后再强制作答 1 次）",
        "tools_t": "平均工具调用次数", "tools_s": "搜索 / grep / 读文件等调用总数 · 智能体一轮可并行调用多个工具",
        "js": {"docs": "篇", "acc": "准确率", "full": "找全证据", "fr": "误拒", "ro": "说不知道", "per_q": "/题",
               "generated": "生成于", "judged": "已评分", "incomplete": "（数据未完整，空缺处不显示）",
               "rec_names": {"best": "质量首选", "value": "性价比", "flash": "Flash 级首选"},
               "rec_acc": "准确", "rec_ref": "拒答", "nodata": "数据不足", "no_graded": "暂无已评分数据",
               "f_model": "模型", "f_all": "全部", "running": "进行中", "core_h": "核心指标", "label_w": 150,
               "steps": "轮次", "tools": "工具调用", "cap": "用满步数", "better": "更好", "higher": "越高越好", "lower": "越低越好", "f_scale": "知识库体量", "reset": "重置",
               "shown": "显示 {n} 个组合", "no_match": "没有符合筛选条件的组合",
               "fg": {"model": "模型", "mode": "模式", "method": "检索方法", "wiki": "知识库形态"},
               "fv": {"mode": {"A0": "单次", "A2": "两轮", "A1": "智能体"},
                      "method": {"grep": "grep", "bm25": "BM25", "hybrid": "混合", "enhanced": "增强混合", "nav": "wiki 导航"},
                      "wiki": {"raw": "原文", "wiki": "LLM-wiki"}},
               "th": ["配置", "模型", "准确率", "找全证据", "误拒", "说不知道", "轮次", "工具调用", "p50", "$/题", "已评分"]},
    },
    "en": {
        "lang": "en", "other_href": "report.html", "other_lang": "zh", "other_label": "中文", "title": "Knowledge-base retrieval benchmark", "h1": "Knowledge-base retrieval benchmark · core metrics",
        "recs_h": "Recommendations", "core_h": "Core metrics",
        "acc_t": "MuSiQue accuracy",
        "acc_s": "Higher is better · multi-hop QA, short answers · questions any model answers without the KB are excluded",
        "all_t": "Full evidence found", "all_s": "Higher is better · every evidence document of a multi-hop question was retrieved or read",
        "fr_t": "False refusal", "fr_s": "Lower is better · says \"not in the KB\" although the answer is there",
        "ro_t": "Says \"not in the KB\"", "ro_s": "Higher is better · correct refusal when the KB lacks the answer",
        "value_h": "Cost and speed", "cost_t": "Accuracy vs cost per question", "cost_s": "Log-scale x axis · top-left is best",
        "lat_t": "Accuracy vs latency (p50)",
        "lat_s": "Top-left is best · measured under concurrent load; compare across API providers with care",
        "table_h": "Data table", "k_w": 104,
        "rounds_h": "Rounds", "steps_t": "Average rounds (model calls)",
        "steps_s": "Fewer is faster and cheaper · single-shot = 1; two-round ≤ 3; agent ≤ 12 rounds (+1 forced answer at the cap)",
        "tools_t": "Average tool calls", "tools_s": "Search / grep / read calls in total · an agent may call several tools per round",
        "js": {"docs": "docs", "acc": "Accuracy", "full": "Full evidence", "fr": "False refusal", "ro": "Says not in KB",
               "per_q": "/question", "generated": "Generated", "judged": "graded",
               "incomplete": " (incomplete — missing cells are hidden)",
               "rec_names": {"best": "Best quality", "value": "Best value", "flash": "Best Flash-tier"},
               "rec_acc": "acc", "rec_ref": "refusal", "nodata": "insufficient data", "no_graded": "No graded data yet",
               "f_model": "Model", "f_all": "All", "running": "running", "core_h": "Core metrics", "label_w": 215,
               "steps": "rounds", "tools": "tool calls", "cap": "hit step cap", "better": "better", "higher": "higher is better", "lower": "lower is better", "f_scale": "KB size", "reset": "Reset",
               "shown": "{n} combinations shown", "no_match": "No combination matches the filters",
               "fg": {"model": "Model", "mode": "Mode", "method": "Retrieval", "wiki": "KB form"},
               "fv": {"mode": {"A0": "Single-shot", "A2": "Two-round", "A1": "Agent"},
                      "method": {"grep": "grep", "bm25": "BM25", "hybrid": "Hybrid", "enhanced": "Enhanced hybrid", "nav": "Wiki navigation"},
                      "wiki": {"raw": "Raw notes", "wiki": "LLM-wiki"}},
               "th": ["Configuration", "Model", "Accuracy", "Full evidence", "False refusal", "Says not in KB", "Rounds",
                      "Tool calls", "p50", "$/question", "Graded"]},
    },
}


def localize(data: dict, lang: str) -> dict:
    """Swap configuration labels for the target language (data is collected with the Chinese labels)."""
    if lang == "zh":
        return data
    tr = lambda s: CONFIG_EN.get(s, s)  # noqa: E731
    out = json.loads(json.dumps(data))
    out["configs"] = [tr(c) for c in out["configs"]]
    for m in out["config_meta"]:
        m["label"] = tr(m["label"])
    for c in out["cells"]:
        c["config"] = tr(c["config"])
    for rec in out["recs"].values():
        for v in rec.values():
            if v:
                v["config"] = tr(v["config"])
    return out


def render_html(data: dict, lang: str) -> str:
    S = STRINGS[lang]
    html = HTML
    for k, v in S.items():
        if k != "js":
            html = html.replace("{{" + k + "}}", str(v))
    html = html.replace("{{appendix}}", appendix_html(lang, MEMORIZED, data["judged"]))
    return (html.replace("__DATA__", json.dumps(localize(data, lang), ensure_ascii=False))
                .replace("__T__", json.dumps(S["js"], ensure_ascii=False)))


def collect() -> dict:
    d = Data()
    MEMORIZED.update(d.memorized)
    cells = []
    for s in SCALES:
        for mode, cfg, label in CONFIGS:
            for m in MODELS:
                c = d.cell(m, mode, cfg, s)
                if not c["n_raw"]:
                    continue
                cov = c["n_raw"] / TOTAL_PER_SCALE[s]
                vals = {k: (round(c[k], 4) if isinstance(c[k], float) else c[k])
                        for k in ("acc", "acc_n", "mus_all", "false_ref", "refusal_ok", "lat50", "cost",
                                  "n_raw", "n_judged", "steps", "tool_calls", "cap_hit")}
                if cov < 0.9:  # still running: partial numbers would mislead
                    vals.update({k: None for k in ("acc", "mus_all", "false_ref", "refusal_ok", "lat50", "cost", "steps", "tool_calls", "cap_hit")})
                elif c["n_judged"] < 0.9 * c["n_raw"]:  # run done, grading not yet: hide grader-based metrics
                    vals.update({k: None for k in ("acc", "false_ref", "refusal_ok")})
                cells.append({"scale": s, "config": label, "model": m, "coverage": round(cov, 3), **vals})
    recs = {}
    for s in SCALES:
        r = recommend(d, s)
        recs[s] = {k: (None if not v else {"model": v[0], "config": v[1], "acc": v[2]["acc"], "cost": v[2]["cost"], "refusal_ok": v[2]["refusal_ok"],
                                           "lat50": v[2]["lat50"]}) for k, v in r.items()} if r else {}
    judged = sum(c["n_judged"] for c in cells)
    expected = sum(TOTAL_PER_SCALE[s] for s in SCALES) * len(CONFIGS) * len(MODELS)
    method = {"R1_grep": "grep", "R2_bm25": "bm25", "R3_hybrid": "hybrid", "R4_enhanced": "enhanced", "R5_wiki_nav": "nav", "R6_wiki_hybrid": "hybrid"}
    meta = [{"label": label, "mode": mode, "method": method[cfg], "wiki": "wiki" if cfg.startswith(("R5", "R6")) else "raw"}
            for mode, cfg, label in CONFIGS]
    return {"cells": cells, "recs": recs, "configs": [c[2] for c in CONFIGS], "config_meta": meta, "models": MODELS,
            "labels": MODEL_LABEL, "scales": SCALES, "judged": judged, "expected": expected,
            "generated": f"{datetime.now():%Y-%m-%d %H:%M}"}


HTML = r"""<!doctype html>
<html lang="{{lang}}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{title}}</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --text-muted:#8a8984;--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
 --surface-2:#242422;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--text-primary:#f0efec;
 --text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 -apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif;
 padding:24px 28px 48px;min-height:100vh}
h1{font-size:20px;margin:0 0 4px}.head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
.links{display:flex;gap:8px;font-size:13px;color:var(--text-muted)}.links a{color:var(--text-secondary);text-decoration:none;border-bottom:1px solid var(--grid)}.links a:hover{color:var(--text-primary);border-color:var(--text-muted)}h2{font-size:15px;margin:28px 0 10px}.meta{color:var(--text-secondary);font-size:12px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:16px 0 4px;position:sticky;top:0;background:var(--surface-1);padding:8px 0;z-index:2}
.seg button{border:1px solid var(--grid);background:var(--surface-2);color:var(--text-primary);padding:5px 12px;cursor:pointer;font:inherit}
.seg button:first-child{border-radius:6px 0 0 6px}.seg button:last-child{border-radius:0 6px 6px 0}
.seg button.on{background:var(--text-primary);color:var(--surface-1);border-color:var(--text-primary)}
.legend{display:flex;gap:14px;color:var(--text-secondary);font-size:13px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.recs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.card{background:var(--surface-2);border-radius:8px;padding:12px 14px}.card b{font-size:13px}.card div{font-size:12.5px;color:var(--text-secondary);margin-top:6px}
.card .k{color:var(--text-muted);display:inline-block;width:{{k_w}}px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 28px}
.panel h3{font-size:13px;margin:0 0 2px}.panel .sub{font-size:12px;color:var(--text-muted);margin-bottom:4px}
svg text{fill:var(--text-secondary);font-size:11.5px}svg .ax{fill:var(--text-muted);font-size:10.5px}svg .better{fill:var(--text-muted);font-size:10.5px}
th .dir{color:var(--text-muted);font-weight:500}#tbl td.m{min-width:66px}#tbl td.best span{font-weight:650;color:var(--text-primary)}
.track{display:block;height:3px;background:var(--grid);border-radius:2px;margin-top:3px;overflow:hidden}.fill{display:block;height:100%;background:var(--text-secondary);opacity:.55;border-radius:2px;margin-left:auto}
.gl{stroke:var(--grid);stroke-width:1}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--grid);border-radius:6px;padding:7px 9px;font-size:12px;
 box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;z-index:9;max-width:280px}#tip .t{font-weight:600;margin-bottom:3px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
#tbl th{position:sticky;top:var(--barh,44px)}th{color:var(--text-secondary);font-weight:600;background:var(--surface-1)}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
td .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
/* rows tinted by model; hover deepens; .hl = linked from a chart point */
#tbl tbody tr{--rowc:var(--s1);cursor:default}
#tbl tbody tr.m-haiku{--rowc:var(--s1)}#tbl tbody tr.m-sonnet{--rowc:var(--s2)}#tbl tbody tr.m-deepseek{--rowc:var(--s3)}
#tbl tbody td{background:color-mix(in srgb,var(--rowc) 7%,transparent);transition:background-color .12s}
#tbl tbody tr:hover td{background:color-mix(in srgb,var(--rowc) 17%,transparent)}
#tbl tbody tr.hl td{background:color-mix(in srgb,var(--rowc) 28%,transparent)}
#tbl tbody tr.hl td:first-child,#tbl tbody tr:hover td:first-child{box-shadow:inset 3px 0 0 var(--rowc)}
/* global filter bar */
.bar .grp{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.bar .lbl{color:var(--text-muted);font-size:12px;margin-right:2px}
.bar .shown{color:var(--text-muted);font-size:12px}.bar .reset{border:none;background:none;color:var(--text-secondary);font:inherit;font-size:12.5px;cursor:pointer;text-decoration:underline;padding:0}
.nomatch{color:var(--text-muted);font-size:12.5px;padding:18px 0}
/* table model filter: toggle chips */
.tfilter{display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap}
.tfilter .lbl{color:var(--text-muted);font-size:12px;margin-right:2px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--grid);background:var(--surface-2);color:var(--text-primary);
 border-radius:999px;padding:4px 11px;font:inherit;font-size:12.5px;cursor:pointer;transition:opacity .12s,background-color .12s}
.chip i{width:9px;height:9px;border-radius:50%;display:inline-block}
.chip[aria-pressed="false"]{opacity:.45;background:transparent}
.chip[aria-pressed="false"] i{background:transparent!important;box-shadow:inset 0 0 0 1.5px var(--text-muted)}
.chip:hover{border-color:var(--text-muted)}
/* chart points: linked highlight from table rows / other charts */
.pt{cursor:pointer}.pt .mark{transition:r .12s,stroke .12s}
.pt.hl .mark{r:8px;stroke:var(--text-primary);stroke-width:2px}
/* appendix */
.appx{margin-top:40px;border-top:1px solid var(--grid);padding-top:8px;font-size:13px;color:var(--text-secondary)}
.appx h3{font-size:14px;color:var(--text-primary);margin:22px 0 8px}.appx h4{font-size:13px;color:var(--text-primary);margin:14px 0 6px}
.appx table{font-size:12.5px;margin:4px 0 8px}.appx td,.appx th{white-space:normal;text-align:left;vertical-align:top}
.appx td.num,.appx th.num{text-align:right;white-space:nowrap}.appx ul{margin:4px 0 8px;padding-left:20px}.appx li{margin:2px 0}
.appx a{color:var(--s1)}.appx code{font-size:12px;background:var(--surface-2);padding:0 4px;border-radius:3px}
@media (max-width:900px){.grid2,.recs{grid-template-columns:1fr}}
</style></head>
<body><div class="viz-root">
<div class="head"><h1>{{h1}}</h1><nav class="links"><a href="{{other_href}}" hreflang="{{other_lang}}">{{other_label}}</a><span>·</span><a href="https://github.com/leweii/kb-retrieval-benchmark" target="_blank" rel="noopener">GitHub</a></nav></div>
<div class="meta" id="meta"></div>
<div class="bar" id="filters"></div>
<h2>{{recs_h}}</h2><div class="recs" id="recs"></div>
<h2 id="h-core">{{core_h}}</h2>
<div class="grid2">
 <div class="panel"><h3>{{acc_t}}</h3><div class="sub">{{acc_s}}</div><div id="p-acc"></div></div>
 <div class="panel"><h3>{{all_t}}</h3><div class="sub">{{all_s}}</div><div id="p-all"></div></div>
 <div class="panel"><h3>{{fr_t}}</h3><div class="sub">{{fr_s}}</div><div id="p-fr"></div></div>
 <div class="panel"><h3>{{ro_t}}</h3><div class="sub">{{ro_s}}</div><div id="p-ro"></div></div>
</div>
<h2>{{rounds_h}}</h2>
<div class="grid2">
 <div class="panel"><h3>{{steps_t}}</h3><div class="sub">{{steps_s}}</div><div id="p-steps"></div></div>
 <div class="panel"><h3>{{tools_t}}</h3><div class="sub">{{tools_s}}</div><div id="p-tools"></div></div>
</div>
<h2>{{value_h}}</h2>
<div class="grid2">
 <div class="panel"><h3>{{cost_t}}</h3><div class="sub">{{cost_s}}</div><div id="p-cost"></div></div>
 <div class="panel"><h3>{{lat_t}}</h3><div class="sub">{{lat_s}}</div><div id="p-lat"></div></div>
</div>
<h2>{{table_h}}</h2><div id="tbl"></div>
{{appendix}}
<div id="tip"></div>
</div>
<script>
const D = __DATA__;
const T = __T__;
const COL = {haiku:"var(--s1)", sonnet:"var(--s2)", deepseek:"var(--s3)"};
const pct = v => v==null ? "–" : Math.round(v*100)+"%";
const usd = v => v==null ? "–" : "$"+(v<0.01 ? v.toFixed(4) : v.toFixed(3));
const sec = v => v==null ? "–" : v.toFixed(v<10?1:0)+"s";
const num = v => v==null ? "–" : v.toFixed(1);
let scale = D.scales[D.scales.length-1];
const tip = document.getElementById("tip");
function showTip(e, html){tip.innerHTML=html;tip.style.display="block";const x=Math.min(e.clientX+14, innerWidth-300);tip.style.left=x+"px";tip.style.top=(e.clientY+14)+"px";}
function hideTip(){tip.style.display="none";}
function cell(c){return `<div class="t">${c.config}</div>${D.labels[c.model]} · ${c.scale} ${T.docs}<br>${T.acc} ${pct(c.acc)} (n=${c.acc_n}) · ${T.full} ${pct(c.mus_all)}<br>${T.fr} ${pct(c.false_ref)} · ${T.ro} ${pct(c.refusal_ok)}<br>${T.steps} ${num(c.steps)} · ${T.tools} ${num(c.tool_calls)}${c.cap_hit!=null?` · ${T.cap} ${pct(c.cap_hit)}`:""}<br>p50 ${sec(c.lat50)} · ${usd(c.cost)}${T.per_q}`;}
const NS="http://www.w3.org/2000/svg";
function el(tag, attrs, parent){const n=document.createElementNS(NS,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(parent)parent.appendChild(n);return n;}
const keyOf = c => `${c.scale}|${c.config}|${c.model}`;
// +1 = higher is better, -1 = lower is better
const DIR = {acc:1, mus_all:1, refusal_ok:1, false_ref:-1, steps:-1, tool_calls:-1, lat50:-1, cost:-1};
let gid = 0;
// neutral (ink, not status colour) band that darkens toward the better end of an axis; colour stays reserved for models
function betterBand(svg, x0, x1, y0, y1, dir){
  const id="bz"+(gid++), defs=el("defs",{},svg), g=el("linearGradient",{id,x1:"0",x2:"1",y1:"0",y2:"0"},defs);
  el("stop",{offset:"0",style:`stop-color:var(--text-primary);stop-opacity:${dir>0?0:0.085}`},g);
  el("stop",{offset:"1",style:`stop-color:var(--text-primary);stop-opacity:${dir>0?0.085:0}`},g);
  el("rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,fill:`url(#${id})`},svg);
  const t=el("text",{x:dir>0?x1:x0,y:y0-5,"text-anchor":dir>0?"end":"start",class:"better"},svg);
  t.textContent=dir>0?`${T.better} →`:`← ${T.better}`;
}
// radial band in the top-left corner of accuracy-vs-cost/latency scatters (high accuracy, low cost/latency)
function cornerBand(svg, x0, x1, y0, y1){
  const id="bz"+(gid++), defs=el("defs",{},svg), g=el("radialGradient",{id,cx:"0",cy:"0",r:"1"},defs);
  el("stop",{offset:"0",style:"stop-color:var(--text-primary);stop-opacity:0.1"},g);
  el("stop",{offset:"1",style:"stop-color:var(--text-primary);stop-opacity:0"},g);
  el("rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,fill:`url(#${id})`},svg);
  const t=el("text",{x:x0+6,y:y0+13,"text-anchor":"start",class:"better"},svg); t.textContent=`↖ ${T.better}`;
}
// linked highlight: the table row and every chart point of one (scale, config, model) light up together
function highlight(key, on){
  document.querySelectorAll(`[data-key="${CSS.escape(key)}"]`).forEach(n=>n.classList.toggle("hl", on));
}
// one chart point: invisible hit target + visible mark; hover = tooltip + linked highlight, click = jump to its row
function point(svg, c, cx, cy){
  const g=el("g",{class:"pt","data-key":keyOf(c)},svg);
  el("circle",{cx,cy,r:12,fill:"transparent"},g);
  el("circle",{cx,cy,r:5,class:"mark",fill:COL[c.model],stroke:"var(--surface-1)","stroke-width":2},g);
  g.addEventListener("mouseenter",()=>highlight(keyOf(c),true));
  g.addEventListener("mousemove",e=>showTip(e,cell(c)));
  g.addEventListener("mouseleave",()=>{hideTip();highlight(keyOf(c),false);});
  g.addEventListener("click",()=>{const row=document.querySelector(`tr[data-key="${CSS.escape(keyOf(c))}"]`);if(row)row.scrollIntoView({behavior:"smooth",block:"center"});});
}

document.getElementById("meta").textContent =
  `${T.generated} ${D.generated} · ${T.judged} ${D.judged} / ${D.expected}` + (D.judged < D.expected*0.98 ? T.incomplete : "");
// ---- global filters: KB size (single), model / mode / retrieval method / KB form (multi, at least one each)
const META=D.config_meta;
const GROUPS=[["model",D.models],["mode",["A0","A2","A1"]],["method",["grep","bm25","hybrid","enhanced","nav"]],["wiki",["raw","wiki"]]];
const F={}; function resetF(){F.scale=D.scales[D.scales.length-1]; GROUPS.forEach(([g,vals])=>F[g]=new Set(vals));} resetF();
const visCfg=()=>META.filter(c=>F.mode.has(c.mode)&&F.method.has(c.method)&&F.wiki.has(c.wiki)).map(c=>c.label);
function visCells(){const vc=new Set(visCfg()); return D.cells.filter(c=>c.scale===F.scale&&F.model.has(c.model)&&vc.has(c.config));}
function filters(){
  const seg=`<div class="grp"><span class="lbl">${T.f_scale}</span><div class="seg">${D.scales.map(s=>`<button data-scale="${s}" class="${s===F.scale?"on":""}">${s} ${T.docs}</button>`).join("")}</div></div>`;
  const grp=GROUPS.map(([g,vals])=>`<div class="grp"><span class="lbl">${T.fg[g]}</span>${vals.map(v=>`<button class="chip" data-g="${g}" data-v="${v}" aria-pressed="${F[g].has(v)}">${g==="model"?`<i style="background:${COL[v]}"></i>${D.labels[v]}`:T.fv[g][v]}</button>`).join("")}</div>`).join("");
  const n=visCfg().length*F.model.size;
  document.getElementById("filters").innerHTML=seg+grp+`<span class="shown">${T.shown.replace("{n}",n)}</span><button class="reset">${T.reset}</button>`;
  document.documentElement.style.setProperty("--barh", document.getElementById("filters").offsetHeight+"px");
}
document.getElementById("filters").addEventListener("click",e=>{
  const sb=e.target.closest("[data-scale]"), ch=e.target.closest(".chip"), rs=e.target.closest(".reset");
  if(sb) F.scale=+sb.dataset.scale;
  else if(ch){const set=F[ch.dataset.g], v=ch.dataset.v; if(set.has(v)){ if(set.size>1) set.delete(v); } else set.add(v);}  // keep >= 1
  else if(rs) resetF(); else return;
  render();
});

function recs(){
  const names=T.rec_names;
  document.getElementById("recs").innerHTML = D.scales.map(s=>{
    const r=D.recs[s]||{};
    const rows=Object.keys(names).map(k=>{const v=r[k];return `<div><span class="k">${names[k]}</span>${v?`${D.labels[v.model]} + ${v.config} · ${T.rec_acc} ${pct(v.acc)} · ${T.rec_ref} ${pct(v.refusal_ok)} · ${usd(v.cost)}`:T.nodata}</div>`;}).join("");
    return `<div class="card"><b>${s} ${T.docs}</b>${rows}</div>`;}).join("");
}

function dotplot(id, metric, maxv){  // maxv: count axis (0..maxv); default = percentage axis
  const host=document.getElementById(id); host.innerHTML="";
  const cfgs=visCfg(), vis=visCells();
  if(!cfgs.length){host.innerHTML=`<div class="nomatch">${T.no_match}</div>`; return;}
  const W=host.clientWidth||520, L=T.label_w, R=14, rowH=28, TP=20, H=TP+cfgs.length*rowH+22;
  const svg=el("svg",{width:W,height:H,role:"img"},host);
  const span=maxv||1, x=v=>L+(v/span)*(W-L-R);
  betterBand(svg, L, W-R, TP, H-20, DIR[metric]);
  [0,.25,.5,.75,1].forEach(t=>{el("line",{x1:x(t*span),x2:x(t*span),y1:TP,y2:H-20,class:"gl"},svg);const tx=el("text",{x:x(t*span),y:H-6,"text-anchor":"middle",class:"ax"},svg);tx.textContent=maxv?(t*span).toFixed(t*span%1?1:0):Math.round(t*100)+"%";});
  cfgs.forEach((cfg,i)=>{
    const y=TP+i*rowH+rowH/2;
    const lb=el("text",{x:L-8,y:y+4,"text-anchor":"end"},svg); lb.textContent=cfg;
    const cs=vis.filter(c=>c.config===cfg&&c[metric]!=null);
    if(cs.length>1){const xs=cs.map(c=>x(c[metric]));el("line",{x1:Math.min(...xs),x2:Math.max(...xs),y1:y,y2:y,stroke:"var(--grid)","stroke-width":2},svg);}
    // per-model vertical offset: identical values (single-shot recall is model-independent) stay visible
    cs.forEach(c=>point(svg, c, x(c[metric]), y+(D.models.indexOf(c.model)-1)*6));
  });
}

function scatter(id, xkey, logx, xfmt){
  const host=document.getElementById(id); host.innerHTML="";
  const W=host.clientWidth||520, H=300, L=44, R=16, TP=10, B=34;
  const svg=el("svg",{width:W,height:H,role:"img"},host);
  const cs=visCells().filter(c=>c.acc!=null&&c[xkey]!=null);
  if(!cs.length){const t=el("text",{x:W/2,y:H/2,"text-anchor":"middle"},svg);t.textContent=visCfg().length&&F.model.size?T.no_graded:T.no_match;return;}
  const xv=c=>logx?Math.log10(c[xkey]):c[xkey];
  let lo=Math.min(...cs.map(xv)), hi=Math.max(...cs.map(xv)); if(hi-lo<1e-9){lo-=1;hi+=1;}
  const pad=(hi-lo)*.06; lo-=pad; hi+=pad; if(!logx) lo=Math.max(0,lo);  // latency axis never below 0
  const x=v=>L+(v-lo)/(hi-lo)*(W-L-R), y=v=>TP+(1-v)*(H-TP-B);
  cornerBand(svg, L, W-R, TP, H-B);
  [0,.25,.5,.75,1].forEach(t=>{el("line",{x1:L,x2:W-R,y1:y(t),y2:y(t),class:"gl"},svg);const tx=el("text",{x:L-6,y:y(t)+4,"text-anchor":"end",class:"ax"},svg);tx.textContent=Math.round(t*100)+"%";});
  const ticks=[]; if(logx){for(let p=Math.ceil(lo);p<=Math.floor(hi);p++)ticks.push(p);} else {const st=(hi-lo)/4;for(let i=0;i<=4;i++)ticks.push(lo+st*i);}
  ticks.forEach(t=>{const tx=el("text",{x:x(t),y:H-12,"text-anchor":"middle",class:"ax"},svg);tx.textContent=xfmt(logx?Math.pow(10,t):t);});
  cs.forEach(c=>point(svg, c, x(xv(c)), y(c.acc)));
}


function table(){
  const cs=visCells();
  const rows=visCfg().flatMap(cfg=>D.models.map(m=>cs.find(c=>c.config===cfg&&c.model===m)).filter(Boolean));
  if(!rows.length){document.getElementById("tbl").innerHTML=`<div class="nomatch">${T.no_match}</div>`; return;}
  // goodness bar per metric cell: longer = better in every column (direction from DIR), relative to the visible rows
  const COLS=[["acc",pct],["mus_all",pct],["false_ref",pct],["refusal_ok",pct],["steps",num],["tool_calls",num],["lat50",sec],["cost",usd]];
  const rng={}; COLS.forEach(([k])=>{const v=rows.map(c=>c[k]).filter(x=>x!=null); rng[k]=[Math.min(...v),Math.max(...v)];});
  const mcell=(c,k,f)=>{const v=c[k]; if(v==null) return `<td class="m">–</td>`; const [lo,hi]=rng[k];
    const g=hi>lo?(DIR[k]>0?(v-lo)/(hi-lo):(hi-v)/(hi-lo)):1;
    return `<td class="m${g>0.999?" best":""}"><span>${f(v)}</span><i class="track"><i class="fill" style="width:${Math.round(g*100)}%"></i></i></td>`;};
  const dirs=[0,0,...COLS.map(([k])=>DIR[k]),0];
  document.getElementById("tbl").innerHTML = `<table><thead><tr>${T.th.map((h,i)=>`<th>${h}${dirs[i]?` <span class="dir" title="${dirs[i]>0?T.higher:T.lower}">${dirs[i]>0?"↑":"↓"}</span>`:""}</th>`).join("")}</tr></thead><tbody>`+
    rows.map(c=>`<tr class="m-${c.model}" data-key="${keyOf(c)}"><td>${c.config}</td><td><span class="dot" style="background:${COL[c.model]}"></span>${D.labels[c.model]}</td>${COLS.map(([k,f])=>mcell(c,k,f)).join("")}<td>${c.coverage<0.9?T.running+" "+Math.round(c.coverage*100)+"%":c.n_judged+"/"+c.n_raw}</td></tr>`).join("")+`</tbody></table>`;
}

// table row hover -> highlight the same entity's points in every chart (event delegation; table is re-rendered)
const tbl=document.getElementById("tbl");
tbl.addEventListener("mouseover",e=>{const tr=e.target.closest("tr[data-key]");if(tr&&!tr.contains(e.relatedTarget))highlight(tr.dataset.key,true);});
tbl.addEventListener("mouseout",e=>{const tr=e.target.closest("tr[data-key]");if(tr&&!tr.contains(e.relatedTarget))highlight(tr.dataset.key,false);});

function render(){
  scale=F.scale; filters();
  document.getElementById("h-core").textContent=`${T.core_h} · ${scale} ${T.docs}`;
  const mx=k=>Math.ceil(Math.max(...D.cells.map(c=>c[k]||0))); recs(); dotplot("p-steps","steps",mx("steps")); dotplot("p-tools","tool_calls",mx("tool_calls")); dotplot("p-acc","acc"); dotplot("p-all","mus_all"); dotplot("p-fr","false_ref"); dotplot("p-ro","refusal_ok");
  scatter("p-cost","cost",true,v=>"$"+(v<0.01?v.toFixed(3):v.toFixed(2))); scatter("p-lat","lat50",false,v=>v.toFixed(0)+"s"); table();
}
render(); addEventListener("resize", render);
</script></body></html>
"""


if __name__ == "__main__":
    data = collect()
    for lang, name in (("zh", "report.html"), ("en", "report.en.html")):
        html = render_html(data, lang)
        assert "{{" not in html, f"unfilled placeholder in {name}"
        (ROOT / name).write_text(html, encoding="utf-8")
        print(f"wrote {name} ({len(data['cells'])} cells, judged {data['judged']}/{data['expected']})")
