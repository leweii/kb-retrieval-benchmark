"""Export the v2 question set as questions.html (filterable list) and questions.csv.

  uv run python -I scripts/export_questions.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAT = {"musique_ans": "MuSiQue 可回答", "musique_unans": "MuSiQue 不可回答", "bright_ans": "BRIGHT 可回答",
       "bright_unans": "BRIGHT 不可回答", "nolima": "NoLiMa"}


def rows() -> list[dict]:
    qs = [json.loads(l) for l in (ROOT / "questions_v2/questions.jsonl").read_text().splitlines()]
    order = list(CAT)
    qs.sort(key=lambda q: (order.index(q["category"]), q["tier"] != "core", q["qid"]))
    out = []
    for i, q in enumerate(qs, 1):
        if q["answerable"]:
            answer = q["answer"] if q["track"] != "bright" else "（论坛采纳答案，长文，不用于判分）"
        else:
            answer = "知识库中没有（原答案：" + str(q.get("original_answer") or "")[:80].replace("\n", " ") + "）"
        out.append({
            "no": i, "qid": q["qid"], "category": CAT[q["category"]], "tier": "核心" if q["tier"] == "core" else "扩展",
            "scales": "/".join(map(str, q.get("valid_scales", q["scales"]))) if not q["excluded"] else "—",
            "lang": "中文" if q["lang"] == "zh" else "英文", "status": "排除" if q["excluded"] else "有效",
            "question": q["question"], "question_en": q["question_en"], "answer": answer,
            "gold_docs": len(q["gold_paths"]), "exclude_reason": q.get("exclude_reason", ""),
        })
    return out


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark 题目清单</title><style>
:root{color-scheme:light;--bg:#fcfcfb;--bg2:#f3f2ef;--fg:#0b0b0b;--fg2:#52514e;--mut:#8a8984;--grid:#e6e5e1;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--bg2:#242422;--fg:#f0efec;--fg2:#c3c2b7;--mut:#8f8e86;--grid:#33332f;--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif;padding:24px 28px 48px}
h1{font-size:20px;margin:0 0 4px}.meta{color:var(--fg2);font-size:12px;margin-bottom:14px}
.bar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}
.grp{display:flex;gap:6px;align-items:center}.grp .lbl{color:var(--mut);font-size:12px}
.chip{border:1px solid var(--grid);background:var(--bg2);color:var(--fg);border-radius:999px;padding:3px 10px;font:inherit;font-size:12.5px;cursor:pointer}
.chip[aria-pressed="false"]{opacity:.45;background:transparent}
input{border:1px solid var(--grid);background:var(--bg2);color:var(--fg);border-radius:6px;padding:5px 9px;font:inherit;width:220px}
.count{color:var(--fg2);font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
th,td{border-bottom:1px solid var(--grid);padding:7px 8px;text-align:left;vertical-align:top}
th{color:var(--fg2);font-weight:600;position:sticky;top:46px;background:var(--bg);white-space:nowrap}
td.n,td.t{white-space:nowrap;color:var(--fg2)}tr:hover td{background:color-mix(in srgb,var(--acc) 7%,transparent)}
.q{max-width:640px}.q .orig{color:var(--mut);font-size:12px;margin-top:3px}
details summary{cursor:pointer;color:var(--fg)}details[open] summary{margin-bottom:4px}
.long{white-space:pre-wrap;color:var(--fg2);font-size:12.5px;max-height:320px;overflow:auto}
.ans{max-width:260px;color:var(--fg2)}.ex td{opacity:.55}.tag{font-size:11px;padding:1px 6px;border-radius:4px;background:var(--bg2);color:var(--fg2)}
</style></head><body>
<h1>Benchmark 题目清单</h1>
<div class="meta">共 __TOTAL__ 道（有效 __LIVE__，排除 __EXCL__）· 核心题在 300 / 900 / 3000 篇都问，扩展题只在 900 / 3000 篇问 · 中文题由英文原题翻译，原文显示在下方灰字</div>
<div class="bar">
 <div class="grp"><span class="lbl">题组</span><span id="f-cat"></span></div>
 <div class="grp"><span class="lbl">语言</span><span id="f-lang"></span></div>
 <div class="grp"><span class="lbl">状态</span><span id="f-status"></span></div>
 <input id="q" placeholder="搜索题目 / 答案"><span class="count" id="count"></span>
</div>
<table><thead><tr><th>#</th><th>题组</th><th>层级</th><th>问的体量</th><th>语言</th><th>题目</th><th>标准答案</th><th>证据篇数</th></tr></thead><tbody id="tb"></tbody></table>
<script>
const R=__ROWS__;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const F={category:new Set(),lang:new Set(),status:new Set(["有效"])};
function chips(id,key,vals){const h=document.getElementById(id);h.innerHTML=vals.map(v=>`<button class="chip" data-k="${key}" data-v="${esc(v)}" aria-pressed="${!F[key].size||F[key].has(v)}">${esc(v)}</button>`).join(" ");}
function uniq(k){return [...new Set(R.map(r=>r[k]))];}
function cellQ(r){const long=r.question.length>220;const main=long?`<details><summary>${esc(r.question.slice(0,160))}…</summary><div class="long">${esc(r.question)}</div></details>`:esc(r.question);
 const orig=r.lang==="中文"?`<div class="orig">${esc(r.question_en.length>220?r.question_en.slice(0,220)+"…":r.question_en)}</div>`:"";return main+orig;}
function render(){
 const term=document.getElementById("q").value.trim().toLowerCase();
 const rs=R.filter(r=>(!F.category.size||F.category.has(r.category))&&(!F.lang.size||F.lang.has(r.lang))&&(!F.status.size||F.status.has(r.status))
   &&(!term||(r.question+" "+r.question_en+" "+r.answer).toLowerCase().includes(term)));
 document.getElementById("count").textContent=`显示 ${rs.length} 道`;
 document.getElementById("tb").innerHTML=rs.map(r=>`<tr class="${r.status==="排除"?"ex":""}"><td class="n">${r.no}</td><td class="t">${esc(r.category)}${r.status==="排除"?' <span class="tag" title="'+esc(r.exclude_reason)+'">排除</span>':""}</td><td class="t">${r.tier}</td><td class="t">${r.scales}</td><td class="t">${r.lang}</td><td class="q">${cellQ(r)}</td><td class="ans">${esc(r.answer)}</td><td class="t">${r.gold_docs}</td></tr>`).join("");
 chips("f-cat","category",uniq("category"));chips("f-lang","lang",uniq("lang"));chips("f-status","status",uniq("status"));
}
document.body.addEventListener("click",e=>{const b=e.target.closest(".chip");if(!b)return;const s=F[b.dataset.k],v=b.dataset.v;
 if(!s.size){uniq(b.dataset.k==="category"?"category":b.dataset.k).forEach(x=>{if(x!==v)s.add(x)});} else if(s.has(v)){s.delete(v);} else {s.add(v);}
 if(s.size===uniq(b.dataset.k).length)s.clear(); render();});
document.getElementById("q").addEventListener("input",render);render();
</script></body></html>"""


if __name__ == "__main__":
    rs = rows()
    with open(ROOT / "questions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    live = sum(r["status"] == "有效" for r in rs)
    page = (PAGE.replace("__ROWS__", json.dumps(rs, ensure_ascii=False)).replace("__TOTAL__", str(len(rs)))
                .replace("__LIVE__", str(live)).replace("__EXCL__", str(len(rs) - live)))
    (ROOT / "questions.html").write_text(page, encoding="utf-8")
    print(f"wrote questions.html and questions.csv ({len(rs)} questions, {live} live)")
