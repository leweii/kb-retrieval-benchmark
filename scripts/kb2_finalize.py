"""v2 final step: mark unanswerable questions that leaked (kb2_validate.py) as excluded.

Pipeline: build_kb2.py -> translate_questions.py questions_v2/questions.jsonl -> kb2_validate.py -> kb2_finalize.py
Excluded questions stay in the file (excluded=true, exclude_reason) and are skipped by the harness.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QF = ROOT / "questions_v2/questions.jsonl"
val = {r["qid"]: r for r in map(json.loads, (ROOT / "questions_v2/validation.jsonl").read_text().splitlines())}
bscale_path = ROOT / "questions_v2/validation_bright_scales.jsonl"
bscale = {}  # (qid, scale) -> substantive answer possible (kb2_validate_bright.py, grader-aligned criterion)
if bscale_path.exists():
    for r in map(json.loads, bscale_path.read_text().splitlines()):
        bscale[(r["qid"], r["scale"])] = r
qs = [json.loads(l) for l in QF.read_text().splitlines()]
for q in qs:
    v = val.get(q["qid"])
    q["excluded"] = bool(v and v["answerable_from_candidates"])
    q["valid_scales"] = list(q["scales"])
    if q["category"] == "bright_unans":  # unanswerable only where no remaining doc allows a substantive answer
        q["valid_scales"] = [s for s in q["scales"]
                             if not bscale.get((q["qid"], s), {}).get("substantive_answer_possible", False)]
        if not q["valid_scales"]:
            q["excluded"] = True
    if q["excluded"]:
        why = f"leak: {v['supporting_paths']} — {v['evidence'][:300]}" if v and v["answerable_from_candidates"] \
            else "answerable from remaining documents at every scale"
        q["exclude_reason"] = why
    else:
        q.pop("exclude_reason", None)
QF.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
live = [q for q in qs if not q["excluded"]]
print("excluded:", sum(q["excluded"] for q in qs), "| live:", len(live))
print("questions per scale:", {s: sum(s in q["valid_scales"] for q in live) for s in (300, 900, 3000)})
print("bright_unans per scale:", {s: sum(s in q["valid_scales"] for q in live if q["category"] == "bright_unans")
                                  for s in (300, 900, 3000)})
