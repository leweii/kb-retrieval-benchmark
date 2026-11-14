"""Build REPORT.md from P2 results (works on partial data; missing cells are marked).

Inputs (per model m in haiku / sonnet / deepseek):
  results/p2_{m}.jsonl       first batch: A0:R2, A2:R2, A1:R1/R2/R5 (only these configs are taken from it)
  results/p2_{m}_b.jsonl     A0:R3, A2:R3, A1:R3, A1:R6
  results/p2_{m}_r4.jsonl    A0:R4, A1:R4 (Haiku LLM reranker)
  and the matching *.judged.jsonl files (judge.py) for accuracy / refusal metrics.
Raw records give recall, latency, cost, steps; judged records give accuracy and refusal behaviour.
Duplicate (qid, model, mode, config, scale) keys keep the last record.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = ["haiku", "sonnet", "deepseek"]
MODEL_LABEL = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "deepseek": "DeepSeek V4.1 Flash"}
SCALES = [300, 900, 3000]
CONFIGS = [  # (mode, config, label)
    ("A0", "R2_bm25", "单次 · BM25"), ("A0", "R3_hybrid", "单次 · 混合"), ("A0", "R4_enhanced", "单次 · 增强混合(QMD式)"),
    ("A2", "R2_bm25", "两轮 · BM25"), ("A2", "R3_hybrid", "两轮 · 混合"),
    ("A1", "R1_grep", "智能体 · grep"), ("A1", "R2_bm25", "智能体 · BM25"), ("A1", "R3_hybrid", "智能体 · 混合"),
    ("A1", "R4_enhanced", "智能体 · 增强混合"), ("A1", "R5_wiki_nav", "智能体 · LLM-wiki 导航"),
    ("A1", "R6_wiki_hybrid", "智能体 · LLM-wiki+混合"),
]
SOURCES = {  # file suffix -> configs taken from it
    "": {"R1_grep", "R2_bm25", "R5_wiki_nav"},
    "_b": {"R3_hybrid", "R6_wiki_hybrid"},
    "_r4": {"R4_enhanced"},
}
TOTAL_PER_SCALE = {}  # filled from the question file


def key(r: dict) -> tuple:
    return (r["qid"], r["model"], r["mode"], r["config"], r["scale"])


def load(path: Path, configs: set[str]) -> dict[tuple, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["config"] in configs and r.get("stop") != "error":
                out[key(r)] = r
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fmt_pct(x, n=None, ci=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.0%}" + (f" ({n})" if n is not None else "")


def evidence_found(r: dict, gold_paths: list[str], all_: bool = True, opened_only: bool = False) -> bool:
    """Gold documents among those the model was shown: passages (single-shot / two-round), search snippets,
    grep hit lines (one line per hit, <= 300 chars) and files opened with read_file. With opened_only, only
    files opened with read_file count.
    The exposure measure grows with the number of search / grep calls (one grep call can list up to 60 files),
    so it is not a measure of how thoroughly a model searched. The opened-only measure is no better for grep:
    models often answer from the hit lines without opening the file (Sonnet + grep at 3,000 docs: 84% accuracy
    with all evidence opened in 9% of questions). Report both; draw no conclusion from either alone."""
    seen = set(r["read"]) if opened_only else set(r["retrieved"]) | set(r["read"])
    gold = set(gold_paths)
    return gold <= seen if all_ else bool(gold & seen)


def evidence_noise(r: dict, gold_paths: list[str]) -> float | None:
    """Share of the documents shown to the model (same exposure set as evidence_found) that are not evidence —
    1 - precision. Recall alone rewards showing more: 100 documents with 5 needed and 20 with the same 5 both
    find all the evidence, at 95% and 75% noise. None when nothing was shown. Evidence is the minimal gold set, so
    other documents that also hold the answer count as noise (an upper bound; use it to compare setups)."""
    seen = set(r["retrieved"]) | set(r["read"])
    return len(seen - set(gold_paths)) / len(seen) if seen else None


class Data:
    def __init__(self, qfile: Path | None = None, res: Path | None = None):
        """qfile / res override the question file and the results directory (e.g. to compare a snapshot)."""
        qfile, RES = qfile or ROOT / "questions_v2/questions.jsonl", res or globals()["RES"]
        self.qs = {q["qid"]: q for q in map(json.loads, qfile.read_text().splitlines())}
        for s in SCALES:
            TOTAL_PER_SCALE[s] = sum(1 for q in self.qs.values() if not q["excluded"] and s in q.get("valid_scales", q["scales"]))
        self.raw, self.judged = {}, {}
        for m in MODELS:
            for suf, cfgs in SOURCES.items():
                self.raw.update(load(RES / f"p2_{m}{suf}.jsonl", cfgs))
                self.judged.update(load(RES / f"p2_{m}{suf}.judged.jsonl", cfgs))
                if suf == "":  # first-batch judged records may live in a filtered *_main file
                    self.judged.update(load(RES / f"p2_{m}_main.judged.jsonl", cfgs))
        # judged records carry every raw field: without the raw run logs (e.g. only data/results/p2_judged.tar.gz
        # unpacked into results/), fall back to them
        for k, r in self.judged.items():
            self.raw.setdefault(k, r)
        # keep only questions that are live at that scale
        ok = lambda k: k[0] in self.qs and not self.qs[k[0]]["excluded"] and k[4] in self.qs[k[0]].get("valid_scales", self.qs[k[0]]["scales"])  # noqa: E731
        self.raw = {k: v for k, v in self.raw.items() if ok(k)}
        self.judged = {k: v for k, v in self.judged.items() if ok(k)}
        # MuSiQue questions any model answers closed-book (Sonnet ones were filtered at build time; Haiku and
        # DeepSeek were checked afterwards) are left out of accuracy so memory doesn't count as retrieval.
        self.memorized = set()
        for f in (ROOT / "report").glob("closed_book_*_musique.json"):
            self.memorized |= {r["qid"] for r in json.loads(f.read_text()) if r["correct"]}
        self.g_raw, self.g_jud = defaultdict(list), defaultdict(list)
        for k, r in self.raw.items():
            self.g_raw[k[1:]].append(r)
        for k, r in self.judged.items():
            self.g_jud[k[1:]].append(r)

    def cell(self, model: str, mode: str, config: str, scale: int, lang: str | None = None) -> dict:
        g = (model, mode, config, scale)
        raw = [r for r in self.g_raw[g] if not lang or r["lang"] == lang]
        jud = [r for r in self.g_jud[g] if not lang or r["lang"] == lang]
        q = self.qs

        def gold_found(r, all_=True, opened_only=False):
            return evidence_found(r, q[r["qid"]]["gold_paths"], all_, opened_only)

        mus = [r for r in raw if r["category"] == "musique_ans"]
        bri = [r for r in raw if r["category"] == "bright_ans"]
        nol = [r for r in raw if r["category"] == "nolima"]
        # Scoring rule: a run without a usable response (error / empty answer; the judge marks it null) is a
        # failure on every metric — wrong for accuracy, a non-answer for false refusals, not a correct refusal.
        jm = [r for r in jud if r["category"] == "musique_ans" and r["qid"] not in self.memorized]
        jans = [r for r in jud if r["category"] in ("musique_ans", "bright_ans")]
        jun = [r for r in jud if not q[r["qid"]]["answerable"]]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        return {
            "n_raw": len(raw), "n_judged": len(jud),
            "acc": mean([bool(r["judge"]["correct"]) for r in jm]), "acc_n": len(jm),
            "acc_k": sum(bool(r["judge"]["correct"]) for r in jm),
            "mus_all": mean([gold_found(r) for r in mus]),
            "mus_opened": mean([gold_found(r, opened_only=True) for r in mus]) if mode != "A0" else None,
            "noise": mean([x for x in (evidence_noise(r, q[r["qid"]]["gold_paths"]) for r in mus) if x is not None]),
            "bright_any": mean([gold_found(r, False) for r in bri]),
            "nolima_found": mean([gold_found(r, False) for r in nol]),
            "false_ref": mean([r["judge"]["abstained"] is not False for r in jans]),
            "refusal_ok": mean([r["judge"]["abstained"] is True for r in jun]), "refusal_n": len(jun),
            "no_response": sum(r["judge"]["abstained"] is None for r in jud),
            "lat50": statistics.median([r["latency"] for r in raw]) if raw else None,
            "lat95": sorted(r["latency"] for r in raw)[int(0.95 * (len(raw) - 1))] if raw else None,
            "cost": mean([r["cost"] for r in raw]),
            "steps": mean([r["steps"] for r in raw]),
            "tool_calls": mean([r["tool_calls"] for r in raw]),
            "cap_hit": mean([r["steps"] >= 13 for r in raw]) if mode == "A1" else None,  # 12-step cap + forced answer
            "raw_drill": mean([bool(r.get("read_raw")) for r in raw]) if config.startswith(("R5", "R6")) else None,
        }


def recommend(d: Data, scale: int, min_refusal: float = 0.7, min_n: int = 30, margin: float = 0.05) -> dict:
    """Quality pick: highest MuSiQue accuracy among cells with >= min_refusal correct refusals and >= min_n judged
    MuSiQue clean questions. Value pick: cheapest cell within `margin` of the quality pick. Flash pick: the value
    rule restricted to Haiku / DeepSeek. Avoid: lowest accuracy.
    Ties are broken deterministically: higher accuracy, then higher correct-refusal rate, then lower cost, then
    model / configuration order."""
    cells = []
    for m in MODELS:
        for mode, cfg, label in CONFIGS:
            c = d.cell(m, mode, cfg, scale)
            if c["acc"] is not None and c["acc_n"] >= min_n and (c["refusal_ok"] or 0) >= min_refusal:
                cells.append((m, label, c))
    if not cells:
        return {}
    quality = lambda x: (-x[2]["acc"], -(x[2]["refusal_ok"] or 0), x[2]["cost"] or 0)  # noqa: E731
    cheap = lambda x: (x[2]["cost"] or 0, -x[2]["acc"], -(x[2]["refusal_ok"] or 0))  # noqa: E731
    best = min(cells, key=quality)
    value = min([x for x in cells if x[2]["acc"] >= best[2]["acc"] - margin - 1e-9], key=cheap)
    flash = [x for x in cells if x[0] in ("haiku", "deepseek")]
    fvalue = None
    if flash:
        fbest = min(flash, key=quality)
        fvalue = min([x for x in flash if x[2]["acc"] >= fbest[2]["acc"] - margin - 1e-9], key=cheap)
    worst = max(cells, key=quality)
    return {"best": best, "value": value, "flash": fvalue, "worst": worst}


def desc(x) -> str:
    if not x:
        return "（数据不足）"
    m, label, c = x
    return f"{MODEL_LABEL[m]} + {label}<br>准确率 {c['acc']:.0%}，p50 {c['lat50']:.0f}s，${c['cost']:.3f}/题"


def build() -> str:
    d = Data()
    L = []
    complete = all(d.cell(m, mo, cf, s)["n_judged"] >= TOTAL_PER_SCALE[s] * 0.98
                   for m in MODELS for mo, cf, _ in CONFIGS for s in SCALES)
    L.append(f"# 知识库方案评测报告{'' if complete else '（预览：数据未完整）'}")
    L.append(f"\n生成时间：{datetime.now():%Y-%m-%d %H:%M} · 题库 v2（MuSiQue + BRIGHT + NoLiMa），有效题 "
             f"S300 {TOTAL_PER_SCALE[300]} / S900 {TOTAL_PER_SCALE[900]} / S3000 {TOTAL_PER_SCALE[3000]}")
    if not complete:
        L.append("\n> ⚠️ 预览版：部分配置尚未跑完或尚未评分（见 §1 数据覆盖）。结论和数字会在全部数据到齐后更新。")

    L.append("\n## 0. 结论速览\n")
    L.append("| 体量 | 质量首选 | 性价比首选（与首选差距 ≤5 个百分点中最便宜） | Flash 级首选 | 不推荐 |")
    L.append("|---|---|---|---|---|")
    for s in SCALES:
        r = recommend(d, s)
        L.append(f"| {s} 篇 | {desc(r.get('best'))} | {desc(r.get('value'))} | {desc(r.get('flash'))} | {desc(r.get('worst'))} |")
    L.append("\n选择规则：只在\"不可回答题正确拒答率 ≥ 70%、MuSiQue 已评分 ≥ 30 题\"的组合中比较；质量看 MuSiQue 答案准确率。")

    L.append("\n## 1. 数据覆盖（已跑完 / 已评分，按体量合计）\n")
    L.append("| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    tot = sum(TOTAL_PER_SCALE.values())
    for mode, cfg, label in CONFIGS:
        row = []
        for m in MODELS:
            nr = sum(d.cell(m, mode, cfg, s)["n_raw"] for s in SCALES)
            nj = sum(d.cell(m, mode, cfg, s)["n_judged"] for s in SCALES)
            row.append(f"{nr}/{tot} · 评 {nj}")
        L.append(f"| {label} | " + " | ".join(row) + " |")

    for s in SCALES:
        L.append(f"\n## 2.{SCALES.index(s) + 1} 总表：{s} 篇\n")
        L.append("每格：MuSiQue 准确率 · 找全证据 · 误拒 · 正确拒答 · p50 耗时 · 单题花费")
        L.append("\n| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
        L.append("|---|" + "---|" * len(MODELS))
        for mode, cfg, label in CONFIGS:
            row = []
            for m in MODELS:
                c = d.cell(m, mode, cfg, s)
                if not c["n_raw"]:
                    row.append("待补")
                    continue
                lat = f"{c['lat50']:.0f}s" if c["lat50"] is not None else "–"
                row.append(f"{fmt_pct(c['acc'])} · {fmt_pct(c['mus_all'])} · {fmt_pct(c['false_ref'])} · "
                           f"{fmt_pct(c['refusal_ok'])} · {lat} · ${c['cost']:.3f}")
            L.append(f"| {label} | " + " | ".join(row) + " |")

    L.append("\n## 3. 维度分析\n")
    L.append("### 3.1 体量的影响（MuSiQue 准确率：300 → 900 → 3000）\n")
    L.append("| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for mode, cfg, label in CONFIGS:
        row = [" → ".join(fmt_pct(d.cell(m, mode, cfg, s)["acc"]) for s in SCALES) for m in MODELS]
        L.append(f"| {label} | " + " | ".join(row) + " |")

    L.append("\n### 3.2 中文 vs 英文提问（S3000，MuSiQue 准确率 中 / 英）\n")
    L.append("| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for mode, cfg, label in CONFIGS:
        row = [f"{fmt_pct(d.cell(m, mode, cfg, 3000, 'zh')['acc'])} / {fmt_pct(d.cell(m, mode, cfg, 3000, 'en')['acc'])}"
               for m in MODELS]
        L.append(f"| {label} | " + " | ".join(row) + " |")

    L.append("\n### 3.3 检索专项（S3000）：BRIGHT 至少找到一篇证据 / NoLiMa 找到植入句\n")
    L.append("| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for mode, cfg, label in CONFIGS:
        row = [f"{fmt_pct(d.cell(m, mode, cfg, 3000)['bright_any'])} / {fmt_pct(d.cell(m, mode, cfg, 3000)['nolima_found'])}"
               for m in MODELS]
        L.append(f"| {label} | " + " | ".join(row) + " |")

    L.append("\n### 3.4 智能体的代价（S3000：平均步数 · p95 耗时 · wiki 回读原文比例）\n")
    L.append("| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for mode, cfg, label in CONFIGS:
        if mode == "A0":
            continue
        row = []
        for m in MODELS:
            c = d.cell(m, mode, cfg, 3000)
            if not c["n_raw"]:
                row.append("待补")
                continue
            drill = f" · 回读 {c['raw_drill']:.0%}" if c["raw_drill"] is not None else ""
            row.append(f"{c['steps']:.1f} 步 · {c['lat95']:.0f}s{drill}")
        L.append(f"| {label} | " + " | ".join(row) + " |")

    L.append("\n## 4. 说明与局限\n")
    L.append("- **准确率只看 MuSiQue**（短答案、可精确判定）。BRIGHT 是开放问题，只评检索与拒答；NoLiMa 只评检索"
             "（模型找到植入句后多数不愿做\"住在旁边→去过\"的推断，准确率反映的是推断意愿）。")
    L.append("- **拒答**：不可回答题按体量判定有效性（BRIGHT 在大库中可能被其他文档覆盖）。正确拒答 = 明确说知识库没有。")
    L.append("- **耗时**：多进程并发运行下测得，Anthropic / DeepSeek 通道之间不宜直接比较绝对值。")
    L.append("- **R4 重排**：P2 中使用 Haiku 5.5 作为 LLM 重排器（OpenRouter 额度不足；16 题对比与 Qwen3-Reranker 召回相同）。")
    L.append("- **千问 Flash 未参与**（OpenRouter 额度）。")
    L.append("- **体量**：S300 只问核心题（135 道），S900 / S3000 问全部题；跨体量对比以同一配置为单位。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    (ROOT / "REPORT.md").write_text(build(), encoding="utf-8")
    print("wrote REPORT.md")
