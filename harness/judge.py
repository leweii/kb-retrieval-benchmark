"""Grade run records with Claude Opus 5.5 (not one of the models under test).

Per record the judge returns
  abstained  the response's conclusion is "not in the knowledge base" (general-knowledge extras that are clearly
             labeled as such don't change that)
  correct    short-answer tracks (musique_ans, nolima): matches the reference answer or an alias; else null
Rules (PLAN §5): unanswerable + abstained = correct refusal; unanswerable + answered = hallucination;
answerable + abstained = false refusal. BRIGHT answers are long-form and only graded for abstention.

  uv run python -I harness/judge.py results/pilot_haiku.jsonl     -> results/pilot_haiku.judged.jsonl
Cached per (record key) in the output file; rerun grades only new records.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
JUDGE, PRICE = "claude-opus-5-5", (4.00, 20.00)
QFILE = ROOT / "questions_v2/questions.jsonl"

SYSTEM = """You grade responses of a knowledge-base assistant. The assistant must answer only from its knowledge \
base and say so when the knowledge base lacks the answer.

Return:
- abstained: true if the response's main conclusion is that the knowledge base does not contain the answer \
(or it cannot answer from it) — even if it then adds information explicitly labeled as general knowledge or \
as not from the knowledge base. false if it presents an answer as if supported by the knowledge base, \
including hedged or partial answers.
- correct: only when a short reference answer is given: true if the response's final answer matches the \
reference or one of its aliases (same entity/value; wording may differ, language may differ), false \
otherwise (an abstention is false). Use null when no short reference answer is given.
- note: one short sentence."""

SCHEMA = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "correct": {"type": ["boolean", "null"]},
        "note": {"type": "string"},
    },
    "required": ["abstained", "correct", "note"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=6)


def key(r: dict) -> str:
    return f"{r['qid']}|{r['model']}|{r['mode']}|{r['config']}|{r['scale']}"


def prompt(r: dict, q: dict) -> str:
    if q["category"] in ("musique_ans", "nolima"):
        ref = f"Reference answer: {q['answer']}" + (f"\nAliases: {', '.join(q['answer_aliases'])}" if q.get("answer_aliases") else "")
    elif q["answerable"]:
        ref = "Reference answer: (long-form; grade abstention only, set correct to null)"
    else:
        ref = "Reference answer: none — the knowledge base does not contain the answer (set correct to null)"
    return f"Question: {q['question']}\n{ref}\n\nResponse:\n{r['answer'][:6000]}"


def grade(r: dict, q: dict) -> dict:
    if r["stop"] == "error" or not r["answer"].strip():
        return {"abstained": None, "correct": None, "note": "no response", "judge_cost": 0.0}
    resp = client.messages.create(model=JUDGE, max_tokens=4000, system=SYSTEM,
                                  output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                                  messages=[{"role": "user", "content": prompt(r, q)}])
    v = json.loads(next(b.text for b in resp.content if b.type == "text"))
    v["judge_cost"] = (resp.usage.input_tokens * PRICE[0] + resp.usage.output_tokens * PRICE[1]) / 1e6
    return v


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = src.with_suffix(".judged.jsonl")
    qs = {q["qid"]: q for q in map(json.loads, QFILE.read_text().splitlines())}
    done = {key(r) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [r for r in map(json.loads, src.read_text().splitlines()) if key(r) not in done and r["qid"] in qs]
    lock = threading.Lock()

    def run(r) -> float:
        v = grade(r, qs[r["qid"]])
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({**r, "judge": v}, ensure_ascii=False) + "\n")
        return v["judge_cost"]

    with ThreadPoolExecutor(16) as pool:
        cost = sum(pool.map(run, todo))
    print(f"judged {len(todo)} records (cached {len(done)}), judge cost ${cost:.2f} -> {out}")
