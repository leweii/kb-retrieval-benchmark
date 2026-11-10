"""Aggregate judged records into the plan's metrics.

  uv run python -I harness/report.py results/*.judged.jsonl [--by lang]

Per (model, mode, config, scale[, extra grouping]):
  recall_any / recall_all   gold docs among retrieved ∪ read (answerable questions with gold docs)
  acc                       correct / answerable short-answer questions (musique_ans, nolima)
  false_refusal             abstained / answerable questions
  refusal_ok / halluc       abstained / answered, over unanswerable questions
  lat_p50 / lat_p95, cost/q, steps, raw_drill (wiki configs: share of runs that opened a raw note)
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def summarize(rs: list[dict], qs: dict) -> dict:
    ans = [r for r in rs if qs[r["qid"]]["answerable"]]
    gold_rs = [r for r in ans if qs[r["qid"]]["gold_paths"]]
    # answer accuracy: MuSiQue only. NoLiMa is graded on retrieval (its accuracy mostly reflects whether a model is
    # willing to infer "lives next to X" -> "has been to X's city"); see nolima_acc for the record.
    short = [r for r in ans if r["category"] == "musique_ans" and r["judge"]["correct"] is not None]
    nol = [r for r in ans if r["category"] == "nolima" and r["judge"]["correct"] is not None]
    unans = [r for r in rs if not qs[r["qid"]]["answerable"] and r["judge"]["abstained"] is not None]
    judged_ans = [r for r in ans if r["judge"]["abstained"] is not None and r["category"] != "nolima"]

    def found(r, mode):
        gold = set(qs[r["qid"]]["gold_paths"])
        seen = set(r["retrieved"]) | set(r["read"])
        return bool(gold & seen) if mode == "any" else gold <= seen

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "n": len(rs),
        "recall_any": mean([found(r, "any") for r in gold_rs]),
        "recall_all": mean([found(r, "all") for r in gold_rs]),
        "acc": mean([r["judge"]["correct"] for r in short]),
        "nolima_acc": mean([r["judge"]["correct"] for r in nol]),
        "false_refusal": mean([r["judge"]["abstained"] for r in judged_ans]),
        "refusal_ok": mean([r["judge"]["abstained"] for r in unans]),
        "halluc": mean([not r["judge"]["abstained"] for r in unans]),
        "lat_p50": round(pct([r["latency"] for r in rs], 0.5), 1),
        "lat_p95": round(pct([r["latency"] for r in rs], 0.95), 1),
        "cost_per_q": round(statistics.mean(r["cost"] for r in rs), 5),
        "steps": round(statistics.mean(r["steps"] for r in rs), 1),
        "raw_drill": mean([bool(r.get("read_raw")) for r in rs]) if rs and rs[0]["config"][:2] in ("R5", "R6") else None,
        "errors": sum(r["stop"] == "error" for r in rs),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    by = None
    if "--by" in args:
        by = args[args.index("--by") + 1]
        args = [a for a in args if a not in ("--by", by)]
    qs = {q["qid"]: q for q in map(json.loads, (ROOT / "questions_v2/questions.jsonl").read_text().splitlines())}
    groups = defaultdict(list)
    for f in args:
        for r in map(json.loads, Path(f).read_text().splitlines()):
            if r["qid"] in qs and not qs[r["qid"]]["excluded"]:
                k = (r["model"], r["mode"], r["config"], r["scale"]) + ((r[by],) if by else ())
                groups[k].append(r)
    cols = ["n", "recall_any", "recall_all", "acc", "nolima_acc", "false_refusal", "refusal_ok", "halluc",
            "lat_p50", "lat_p95", "cost_per_q", "steps", "raw_drill", "errors"]
    head = "model    mode config          scale" + (f" {by:8s}" if by else "") + " " + " ".join(f"{c:>9s}" for c in cols)
    print(head)
    rows = []
    for k in sorted(groups):
        s = summarize(groups[k], qs)
        rows.append({"model": k[0], "mode": k[1], "config": k[2], "scale": k[3], **({by: k[4]} if by else {}), **s})
        cells = " ".join(f"{('-' if s[c] is None else s[c]):>9}" for c in cols)
        print(f"{k[0]:8s} {k[1]:4s} {k[2]:15s} {k[3]:5d}" + (f" {k[4]:8s}" if by else "") + " " + cells)
    (ROOT / "report").mkdir(exist_ok=True)
