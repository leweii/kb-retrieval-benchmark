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
qs = [json.loads(l) for l in QF.read_text().splitlines()]
for q in qs:
    v = val.get(q["qid"])
    q["excluded"] = bool(v and v["answerable_from_candidates"])
    if q["excluded"]:
        q["exclude_reason"] = f"leak: {v['supporting_paths']} — {v['evidence'][:300]}"
    else:
        q.pop("exclude_reason", None)
QF.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
live = [q for q in qs if not q["excluded"]]
print("excluded:", sum(q["excluded"] for q in qs), "| live:", len(live))
print(dict(Counter((q["category"], q["tier"], q["lang"]) for q in live)))
