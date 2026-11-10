"""Closed-book baseline: can a model answer without the knowledge base? (memorization / general-knowledge check)

MuSiQue (answerable, 300 sampled): short answers, scored by normalized match against answer + aliases.
BRIGHT  (StackExchange, 150 sampled): long answers; a judge compares the response with the accepted answer.
Model: Claude Haiku 5.5 (lower bound — stronger models know more). Results -> report/closed_book.json
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
SEED = 21
client = anthropic.Anthropic(max_retries=5)
cost = 0.0


def call(system: str, user: str, max_tokens: int = 4000, schema: dict | None = None) -> str:
    global cost
    oc = {"effort": "low"}
    if schema:
        oc["format"] = {"type": "json_schema", "schema": schema}
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, output_config=oc,
                               messages=[{"role": "user", "content": user}])
    pin, pout = PRICES[MODEL]
    cost += (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
    return "".join(b.text for b in r.content if b.type == "text").strip()


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def musique(rng: random.Random) -> list[dict]:
    rows = [json.loads(l) for l in open(RAW / "musique/musique_full_v1.0_dev.jsonl")]
    rows = rng.sample([r for r in rows if r["answerable"]], 300)
    sys_p = ("Answer the question from your own knowledge with a short phrase only (a name, date, number or place). "
             "If you don't know, answer exactly: unknown")

    def one(r):
        pred = call(sys_p, r["question"], max_tokens=2000)
        golds = [r["answer"]] + r.get("answer_aliases", [])
        hit = any(norm(g) and (norm(g) in norm(pred) or f1(pred, g) >= 0.8) for g in golds)
        return {"id": r["id"], "hops": len(r["question_decomposition"]), "pred": pred, "gold": r["answer"],
                "correct": hit, "abstained": norm(pred) == "unknown"}

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, rows))


JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "partial", "no"]},
                                                 "reason": {"type": "string"}},
                "required": ["verdict", "reason"], "additionalProperties": False}


def bright(rng: random.Random) -> list[dict]:
    rows = []
    for f in sorted((RAW / "bright/examples").glob("*.parquet")):
        for r in pd.read_parquet(f).itertuples():
            rows.append({"id": f"{f.stem}/{r.id}", "q": r.query, "gold": r.gold_answer})
    rows = rng.sample(rows, 150)
    sys_a = "Answer the user's question from your own knowledge in at most 150 words."
    sys_j = ("You compare a response with the accepted answer to a forum question. verdict=yes if the response "
             "conveys the accepted answer's key points, partial if it gets some of them, no otherwise.")

    def one(r):
        pred = call(sys_a, r["q"], max_tokens=3000)
        j = json.loads(call(sys_j, f"Question:\n{r['q'][:4000]}\n\nAccepted answer:\n{r['gold'][:6000]}\n\n"
                                   f"Response:\n{pred}", max_tokens=2000, schema=JUDGE_SCHEMA))
        return {"id": r["id"], "verdict": j["verdict"], "reason": j["reason"]}

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, rows))


if __name__ == "__main__":
    rng = random.Random(SEED)
    m = musique(rng)
    b = bright(rng)
    by_hops = Counter((x["hops"], x["correct"]) for x in m)
    summary = {
        "model": MODEL,
        "musique_closed_book_correct": round(sum(x["correct"] for x in m) / len(m), 3),
        "musique_said_unknown": round(sum(x["abstained"] for x in m) / len(m), 3),
        "musique_correct_by_hops": {h: round(by_hops[(h, True)] / max(1, by_hops[(h, True)] + by_hops[(h, False)]), 2)
                                    for h in sorted({x["hops"] for x in m})},
        "bright_closed_book_verdicts": dict(Counter(x["verdict"] for x in b)),
        "cost_usd": round(cost, 3),
    }
    out = ROOT / "report" / f"closed_book_{MODEL}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "musique": m, "bright": b}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
