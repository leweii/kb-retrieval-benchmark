"""v2 final step: apply the leak checks and the gold audit to questions_v2/questions.jsonl.

Pipeline: build_kb2.py -> translate_questions.py questions_v2/questions.jsonl -> kb2_validate.py
          -> kb2_validate_bright.py -> kb2_leak_check2.py -> kb2_leak_check2_bright.py -> kb2_gold_audit.py
          -> kb2_finalize.py

Inputs (all under questions_v2/):
  validation.jsonl               first leak check (full question, S3000): leaked -> excluded at every scale
  validation_bright_scales.jsonl BRIGHT unanswerable, per scale (first pass, 12k-char truncated documents)
  leak_check2.jsonl              MuSiQue unanswerable, second pass (last-hop retrieval + answer search):
                                 removed from the scales where all supporting documents are present
  leak_check2_bright.jsonl       BRIGHT unanswerable, second pass (chunks, no truncation, documents answering
                                 runs read): removed from the scales where a sufficient document is present
  leak_manual.json               leaks found by hand review (answered-rate flag, review report)
  gold_decisions.json            MuSiQue answerable gold audit after review: drop -> excluded;
                                 add_aliases -> answer_aliases extended (re-judge those records afterwards:
                                 harness/judge.py --regrade)
Excluded questions stay in the file (excluded=true, exclude_reason) and are skipped by the harness and report.
Idempotent: everything is recomputed from the inputs; aliases are only ever added once.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q2 = ROOT / "questions_v2"
QF = Q2 / "questions.jsonl"


def jsonl(name: str) -> list[dict]:
    p = Q2 / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def jsonf(name: str) -> dict:
    p = Q2 / name
    return json.loads(p.read_text()) if p.exists() else {}


def finalize(qs: list[dict], val: dict, bscale: dict, leak2: dict, manual: dict, gold: dict) -> list[dict]:
    """Set excluded / valid_scales / exclude_reason / answer_aliases on every question (in place) from the checks."""
    for q in qs:
        reasons = []
        v = val.get(q["qid"])
        q["excluded"] = bool(v and v["answerable_from_candidates"])
        if q["excluded"]:
            reasons.append(f"leak (check 1): {v['supporting_paths']} — {v['evidence'][:300]}")
        scales = list(q["scales"])
        if q["category"] == "bright_unans":  # unanswerable only where no remaining doc allows a substantive answer
            bad = [s for s in scales if bscale.get((q["qid"], s), {}).get("substantive_answer_possible", False)]
            if bad:
                reasons.append(f"substantive answer possible at {bad} (check 1)")
            scales = [s for s in scales if s not in bad]
        l2 = leak2.get(q["qid"])
        if l2 and l2.get("leaked_scales"):
            bad = [s for s in scales if s in l2["leaked_scales"]]
            if bad:
                paths = l2.get("supporting_paths") or l2.get("sufficient_paths")
                reasons.append(f"leak at {bad} (check 2): {paths} — {l2['evidence'][:300]}")
            scales = [s for s in scales if s not in bad]
        if q["qid"] in manual:
            reasons.append(f"leak (hand review): {manual[q['qid']]['reason']}")
            scales = []
        g = gold.get(q["qid"])
        if g and g["action"] == "drop":
            reasons.append(f"gold audit: {g['defect']} — {g['reason'][:300]}")
            q["excluded"] = True
        if g and g["action"] == "add_aliases":
            have = q.get("answer_aliases") or []
            q["answer_aliases"] = have + [a for a in g["aliases"] if a not in have and a != q["answer"]]
        q["valid_scales"] = scales
        if not scales:
            q["excluded"] = True
        if q["excluded"] or scales != q["scales"]:
            q["exclude_reason"] = " | ".join(reasons)
        else:
            q.pop("exclude_reason", None)
    return qs


if __name__ == "__main__":
    qs = finalize([json.loads(l) for l in QF.read_text().splitlines()],
                  val={r["qid"]: r for r in jsonl("validation.jsonl")},
                  bscale={(r["qid"], r["scale"]): r for r in jsonl("validation_bright_scales.jsonl")},
                  leak2={r["qid"]: r for r in jsonl("leak_check2.jsonl") + jsonl("leak_check2_bright.jsonl")},
                  manual=jsonf("leak_manual.json"), gold=jsonf("gold_decisions.json"))
    QF.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    live = [q for q in qs if not q["excluded"]]
    print("excluded:", sum(q["excluded"] for q in qs), "| live:", len(live))
    print("questions per scale:", {s: sum(s in q["valid_scales"] for q in live) for s in (300, 900, 3000)})
    for cat in ("musique_ans", "musique_unans", "bright_ans", "bright_unans", "nolima"):
        print(f"  {cat}:", {s: sum(s in q["valid_scales"] for q in live if q["category"] == cat) for s in (300, 900, 3000)})
