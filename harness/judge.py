"""Grade run records with Claude Opus 5.5 (not one of the models under test).

Per record the judge returns
  abstained  the response's conclusion is "not in the knowledge base" (general-knowledge extras that are clearly
             labeled as such don't change that)
  correct    short-answer tracks (musique_ans, nolima): matches the reference answer or an alias; else null
Rules (PLAN §5): unanswerable + abstained = correct refusal; unanswerable + answered = hallucination;
answerable + abstained = false refusal. BRIGHT answers are long-form and only graded for abstention.

  uv run python -I harness/judge.py results/pilot_haiku.jsonl     -> results/pilot_haiku.judged.jsonl
Cached per (record key) in the output file; rerun grades only new records.
  --regrade QID,QID,...  re-grade those questions' records (after their reference/aliases changed)
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


def params(r: dict, q: dict) -> dict:
    return dict(model=JUDGE, max_tokens=4000, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt(r, q)}])


def parse(msg, discount: float = 1.0) -> dict:
    v = json.loads(next(b.text for b in msg.content if b.type == "text"))
    v["judge_cost"] = discount * (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) / 1e6
    return v


NO_RESPONSE = {"abstained": None, "correct": None, "note": "no response", "judge_cost": 0.0}


def grade(r: dict, q: dict) -> dict:
    if r["stop"] == "error" or not r["answer"].strip():
        return dict(NO_RESPONSE)
    return parse(client.messages.create(**params(r, q)))


def batch_grade(todo: list[dict], qs: dict, out: Path, write) -> float:
    """Message Batches API (50% price). Batch id + id map are kept in a sidecar file so an interrupted run
    resumes waiting on the same batch instead of resubmitting. Errored/expired items fall back to direct calls."""
    import hashlib
    import time

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    side = out.with_suffix(".batch.json")
    if side.exists():
        meta = json.loads(side.read_text())
    else:
        reqs, ids = [], {}
        for r in todo:
            if r["stop"] == "error" or not r["answer"].strip():
                write(r, dict(NO_RESPONSE))
                continue
            cid = hashlib.sha1(key(r).encode()).hexdigest()[:40]
            ids[cid] = key(r)
            reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params(r, qs[r["qid"]]))))
        if not reqs:
            return 0.0
        b = client.messages.batches.create(requests=reqs)
        meta = {"batch_id": b.id, "ids": ids}
        side.write_text(json.dumps(meta))
        print(f"submitted batch {b.id} with {len(reqs)} requests", flush=True)
    by_key = {key(r): r for r in todo}
    while True:
        b = client.messages.batches.retrieve(meta["batch_id"])
        if b.processing_status == "ended":
            break
        print(f"  batch {b.id}: {b.request_counts.processing} processing, {b.request_counts.succeeded} done", flush=True)
        time.sleep(60)
    cost, retry = 0.0, []
    for res in client.messages.batches.results(meta["batch_id"]):
        k = meta["ids"].get(res.custom_id)
        if k not in by_key:
            continue
        if res.result.type == "succeeded":
            v = parse(res.result.message, discount=0.5)
            cost += v["judge_cost"]
            write(by_key[k], v)
        else:
            retry.append(by_key[k])
    for r in retry:
        v = grade(r, qs[r["qid"]])
        cost += v["judge_cost"]
        write(r, v)
    side.unlink()
    print(f"batch done: {len(meta['ids']) - len(retry)} succeeded, {len(retry)} retried directly", flush=True)
    return cost


if __name__ == "__main__":
    args = sys.argv[1:]
    use_batch = "--batch" in args
    regrade = set()
    if "--regrade" in args:  # comma-separated qids: drop their judged records first, then grade them again
        i = args.index("--regrade")
        regrade = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    src = Path([a for a in args if a != "--batch"][0])
    out = src.with_suffix(".judged.jsonl")
    qs = {q["qid"]: q for q in map(json.loads, QFILE.read_text().splitlines())}
    if regrade and out.exists():
        keep = [l for l in out.read_text().splitlines() if json.loads(l)["qid"] not in regrade]
        print(f"regrade: dropped {len(out.read_text().splitlines()) - len(keep)} judged records")
        out.write_text("".join(l + "\n" for l in keep))
    done = {key(r) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [r for r in map(json.loads, src.read_text().splitlines()) if key(r) not in done and r["qid"] in qs]
    lock = threading.Lock()

    def write(r: dict, v: dict) -> None:
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({**r, "judge": v}, ensure_ascii=False) + "\n")

    if use_batch:
        cost = batch_grade(todo, qs, out, write) if todo else 0.0
    else:
        def run(r) -> float:
            v = grade(r, qs[r["qid"]])
            write(r, v)
            return v["judge_cost"]

        with ThreadPoolExecutor(16) as pool:
            cost = sum(pool.map(run, todo))
    print(f"judged {len(todo)} records (cached {len(done)}), judge cost ${cost:.2f} -> {out}")
