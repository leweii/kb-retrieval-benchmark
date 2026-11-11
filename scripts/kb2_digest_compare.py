"""Compare Haiku vs Sonnet LLM-wiki digests on 50 evidence documents of core questions (blind, judged by Opus 5.5).

For each sampled (question, gold document) pair the judge sees the question + answer, the source document and the
two digests (key facts + summary) in random order, and rates whether each digest keeps the information this
document contributes to answering the question. Output: report/digest_compare.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import digest_wiki as dw  # noqa: E402

JUDGE, JPRICE = "claude-opus-5-5", (4.00, 20.00)
SAMPLE = {"musique": 20, "bright": 20, "nolima": 10}
CORPUS = "kb2/S3000"

SYSTEM = """You evaluate wiki digests of a source document for a knowledge base. You get a question with its \
answer, the source document (one piece of evidence for that question), and two digests X and Y of that \
document. For each digest decide whether it preserves the information that this document contributes to \
answering the question: "yes" (fully), "partial", or "no". Then say which digest is more useful overall for \
finding and understanding this document later ("X", "Y" or "tie")."""

SCHEMA = {
    "type": "object",
    "properties": {
        "x_keeps_needed_info": {"type": "string", "enum": ["yes", "partial", "no"]},
        "y_keeps_needed_info": {"type": "string", "enum": ["yes", "partial", "no"]},
        "better_overall": {"type": "string", "enum": ["X", "Y", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["x_keeps_needed_info", "y_keeps_needed_info", "better_overall", "reason"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)


def render(d: dict) -> str:
    return (f"Summary: {d['summary']}\nKey facts:\n" + "\n".join(f"- {f}" for f in d["key_facts"])
            + "\nEntities: " + "; ".join(f"{e['name']} ({e['role']})" for e in d["entities"]))


def load(model: str, rel: str) -> dict:
    dw.set_model(model)
    p = dw.cache_path(CORPUS, rel)
    if not p.exists():
        dw.digest_one(CORPUS, rel)
    return json.loads(p.read_text())


if __name__ == "__main__":
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl"))
          if q["answerable"] and not q["excluded"] and q["tier"] == "core"]
    rng = random.Random(31)
    pairs = []
    for track, n in SAMPLE.items():
        for q in rng.sample([q for q in qs if q["track"] == track], n):
            pairs.append((q, rng.choice(q["gold_paths"])))

    with ThreadPoolExecutor(12) as pool:  # make sure both digests exist (Sonnet ones may need generating)
        list(pool.map(lambda p: load("sonnet", p[1]), pairs))
    data = [(q, rel, load("sonnet", rel), load("haiku", rel)) for q, rel in pairs]

    def judge(args):
        q, rel, son, hai = args
        flip = random.Random(rel).random() < 0.5
        x, y = (hai, son) if flip else (son, hai)
        src = (dw.corpus_dir(CORPUS) / "raw" / rel).read_text()[:12000]
        ans = q["answer"] if q["track"] != "bright" else q["answer"][:3000]
        user = (f"Question: {q['question_en'][:3000]}\nAnswer: {ans}\n\n<source>\n{src}\n</source>\n\n"
                f"<digest_X>\n{render(x)}\n</digest_X>\n\n<digest_Y>\n{render(y)}\n</digest_Y>")
        r = client.messages.create(model=JUDGE, max_tokens=8000, system=SYSTEM,
                                   output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                                   messages=[{"role": "user", "content": user}])
        v = json.loads(next(b.text for b in r.content if b.type == "text"))
        son_v, hai_v = (v["y_keeps_needed_info"], v["x_keeps_needed_info"]) if flip else (v["x_keeps_needed_info"], v["y_keeps_needed_info"])
        better = {"tie": "tie", "X": "haiku" if flip else "sonnet", "Y": "sonnet" if flip else "haiku"}[v["better_overall"]]
        cost = (r.usage.input_tokens * JPRICE[0] + r.usage.output_tokens * JPRICE[1]) / 1e6
        return {"qid": q["qid"], "track": q["track"], "doc": rel, "sonnet": son_v, "haiku": hai_v,
                "better": better, "reason": v["reason"], "cost": cost}

    with ThreadPoolExecutor(12) as pool:
        res = list(pool.map(judge, data))
    summary = {"by_track": {}, "judge_cost": round(sum(r["cost"] for r in res), 2)}
    for track in SAMPLE:
        rs = [r for r in res if r["track"] == track]
        summary["by_track"][track] = {"sonnet": dict(Counter(r["sonnet"] for r in rs)),
                                      "haiku": dict(Counter(r["haiku"] for r in rs)),
                                      "better": dict(Counter(r["better"] for r in rs))}
    (ROOT / "report/digest_compare.json").write_text(json.dumps({"summary": summary, "items": res}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
