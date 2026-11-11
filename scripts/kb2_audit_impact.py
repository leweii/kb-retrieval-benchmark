"""Before/after comparison of the question-set audit (leak check 2 + gold audit + re-judged aliases).

Compares a snapshot (question file + judged results from before the audit) with the current state, using the
report's own metric code (harness/build_report.py: Data.cell, recommend) so the numbers mean the same thing as
in the report. Adds exact McNemar tests on paired MuSiQue clean-set questions for the key comparisons.

  uv run python -I scripts/kb2_audit_impact.py /tmp/kb_before  -> report/audit_impact.md
  (the snapshot dir holds questions.jsonl and results/p2_*.jsonl / *.judged.jsonl)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.build_report import CONFIGS, MODEL_LABEL, MODELS, SCALES, Data, recommend  # noqa: E402
from harness.stats import COMPARISONS as PAIRS, mcnemar_exact, paired  # noqa: E402

LABEL = {(mode, cfg): label for mode, cfg, label in CONFIGS}
def name(m, mode, cfg) -> str:
    return f"{MODEL_LABEL[m]} · {LABEL[(mode, cfg)]}"


def mcnemar(d: Data, a: tuple, b: tuple, scale: int) -> tuple[int, int, int, float]:
    p = paired(d, a, b, scale)
    bb, cc = sum(x and not y for x, y in p.values()), sum(y and not x for x, y in p.values())
    return len(p), bb, cc, mcnemar_exact(bb, cc)


def pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def rec_str(r) -> str:
    if not r:
        return "（无）"
    m, label, c = r
    return f"{MODEL_LABEL[m]} · {label}（{c['acc']:.0%}，拒答 {c['refusal_ok']:.0%}）"


if __name__ == "__main__":
    snap = Path(sys.argv[1])
    before, after = Data(snap / "questions.jsonl", snap / "results"), Data()
    L = ["# 题库审计前后对比", "",
         "口径与报告一致（harness/build_report.py）：准确率只算 MuSiQue 干净题，没有回答的运行按失败计（报告现行规则，前后两列都用这个规则）。",
         "McNemar 为精确双侧检验，只用两格都有判分的同一批题；b = 前者对后者错，c = 反之。", ""]
    L += ["## 每个体量的有效题数", "", "| 类别 | " + " | ".join(f"S{s} 前→后" for s in SCALES) + " |",
          "|---|" + "---|" * len(SCALES)]
    for cat in ("musique_ans", "musique_unans", "bright_ans", "bright_unans", "nolima"):
        row = []
        for s in SCALES:
            n = [sum(1 for q in d.qs.values() if q["category"] == cat and not q["excluded"]
                     and s in q.get("valid_scales", q["scales"])) for d in (before, after)]
            row.append(f"{n[0]} → {n[1]}")
        L.append(f"| {cat} | " + " | ".join(row) + " |")
    clean = [sum(1 for q in d.qs.values() if q["category"] == "musique_ans" and not q["excluded"]
                 and q["qid"] not in d.memorized) for d in (before, after)]
    L += [f"| MuSiQue 干净题（算准确率） | {clean[0]} → {clean[1]} |  |  |", ""]

    L += ["## 推荐结果", "", "| 体量 | 角色 | 审计前 | 审计后 |", "|---|---|---|---|"]
    for s in SCALES:
        rb, ra = recommend(before, s), recommend(after, s)
        for role, zh in (("best", "质量首选"), ("value", "性价比"), ("flash", "Flash 级")):
            L.append(f"| {s} | {zh} | {rec_str(rb.get(role))} | {rec_str(ra.get(role))} |")
    L.append("")

    for s in SCALES:
        L += [f"## S{s}：准确率与正确拒答（前 → 后）", "",
              "| 配置 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for mode, cfg, label in CONFIGS:
            cells = []
            for m in MODELS:
                cb, ca = before.cell(m, mode, cfg, s), after.cell(m, mode, cfg, s)
                cells.append(f"{pct(cb['acc'])}→**{pct(ca['acc'])}** · 拒 {pct(cb['refusal_ok'])}→{pct(ca['refusal_ok'])}")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L += ["", f"McNemar（S{s}）：", "", "| 比较 | 审计前 n · b/c · p | 审计后 n · b/c · p |", "|---|---|---|"]
        for a, b in PAIRS:
            r = [mcnemar(d, a, b, s) for d in (before, after)]
            L.append(f"| {name(*a)} vs {name(*b)} | " +
                     " | ".join(f"{n} · {bb}/{cc} · p={p:.3f}" for n, bb, cc, p in r) + " |")
        L.append("")
    out = ROOT / "report/audit_impact.md"
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")
