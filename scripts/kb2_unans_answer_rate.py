"""Empirical leak flag: unanswerable questions that most runs answer anyway.

For every live unanswerable question (MuSiQue and BRIGHT) and scale, the share of judged runs (all models and
configurations) that gave an answer instead of saying the knowledge base lacks it. A question that >= FLAG of
runs answer is a leak suspect and is reviewed by hand (the answers and the sources they cite are printed).

  uv run python -I scripts/kb2_unans_answer_rate.py [--show QID]  -> report/unans_answer_rate.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.build_report import Data  # noqa: E402

FLAG = 0.8

if __name__ == "__main__":
    d = Data()
    by = defaultdict(list)
    for (qid, model, mode, config, scale), r in d.judged.items():
        if not d.qs[qid]["answerable"] and r["judge"]["abstained"] is not None:
            by[(qid, scale)].append(r)
    if "--show" in sys.argv:
        qid = sys.argv[sys.argv.index("--show") + 1]
        q = d.qs[qid]
        print(q["question_en"][:400], "\noriginal:", str(q.get("original_answer"))[:300])
        for (k, s), rs in sorted(by.items()):
            if k == qid:
                for r in rs:
                    if not r["judge"]["abstained"]:
                        print(f"\n--- S{s} {r['model']} {r['mode']}:{r['config']}\n{r['answer'][:500]}")
        sys.exit()
    rows = []
    for (qid, scale), rs in by.items():
        rate = sum(not r["judge"]["abstained"] for r in rs) / len(rs)
        per_model = {m: round(sum(not r["judge"]["abstained"] for r in rs if r["model"] == m)
                              / max(1, sum(r["model"] == m for r in rs)), 2) for m in ("haiku", "sonnet", "deepseek")}
        rows.append({"qid": qid, "category": d.qs[qid]["category"], "scale": scale, "runs": len(rs),
                     "answered_rate": round(rate, 3), "per_model": per_model, "flag": rate >= FLAG})
    rows.sort(key=lambda r: -r["answered_rate"])
    (ROOT / "report/unans_answer_rate.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("flagged (qid, scale):", sum(r["flag"] for r in rows), "of", len(rows))
    print("flagged questions by category:", Counter(d.qs[q]["category"] for q in {r["qid"] for r in rows if r["flag"]}))
    for r in rows[:30]:
        print(f"{r['answered_rate']:.2f} S{r['scale']} {r['qid']} {r['per_model']} "
              f"{d.qs[r['qid']]['question_en'][:70]!r}")
