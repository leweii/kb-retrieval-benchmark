"""Run model x retrieval configuration x scale over the v2 question set.

  uv run python -I -m harness.run --model haiku --configs A0:R2_bm25,A1:R1_grep --scales 300,3000 \
      --sample 20 --workers 8 --budget 20 --out results/pilot_haiku.jsonl

  A0:<R>  single shot — retrieve top-8 passages with the question, one model call (R2/R3/R4 only)
  A1:<R>  agent — tool loop over the knowledge base, at most MAX_STEPS model calls
Each finished question is appended to --out immediately (resumable: finished (qid, config, scale, model) are
skipped). Stops scheduling new work when the run's cost reaches --budget (USD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.llm import Conversation  # noqa: E402
from harness.tools import CONFIGS, KB  # noqa: E402

MAX_STEPS = 12
QFILE = ROOT / "questions_v2/questions.jsonl"

BASE_RULES = """- Use only information from the knowledge base. Questions may need facts from several documents.
- If the knowledge base does not contain the information needed to answer, say clearly that it is not in the \
knowledge base (in the user's language). Do not answer from general knowledge.
- Answer in the same language as the question. Give the answer first, concisely.
- End with a line "Sources:" listing the file paths you used."""

SYSTEM_A1 = "You are a knowledge-base assistant. Answer the user's question using the tools to search and read " \
            "the knowledge base.\n" + BASE_RULES
WIKI_HINT = ("\nThe knowledge base is an LLM-maintained wiki. Start by reading index.md. Wiki links like "
             "[[sources/topic/page|Title]] point to files: append .md to the path. Source pages summarize a note and "
             "give its raw note path under raw/ — open it when you need details the summary lacks.")
SYSTEM_A0 = "You are a knowledge-base assistant. Answer the user's question using the retrieved knowledge-base " \
            "passages provided with it.\n" + BASE_RULES


def load_questions(scale: int, sample: int | None) -> list[dict]:
    qs = [q for q in map(json.loads, QFILE.read_text().splitlines()) if not q["excluded"] and scale in q["scales"]]
    if sample:  # stratified by category, deterministic
        by = {}
        for q in sorted(qs, key=lambda q: hashlib.sha1(q["qid"].encode()).hexdigest()):
            by.setdefault(q["category"], []).append(q)
        per = max(1, sample // len(by))
        qs = [q for v in by.values() for q in v[:per]]
    return qs


def run_one(model: str, mode: str, config: str, scale: int, q: dict) -> dict:
    corpus = f"kb2/S{scale}"
    kb = KB(corpus, config)
    t0 = time.time()
    rec = {"qid": q["qid"], "category": q["category"], "lang": q["lang"], "tier": q["tier"], "model": model,
           "mode": mode, "config": config, "scale": scale, "started": t0}
    try:
        if mode == "A0":
            passages = kb.retrieve(q["question"], 8)
            ctx = "\n\n".join(f"<passage path=\"{p['path']}\">\n{p['text']}\n</passage>" for p in passages)
            conv = Conversation(model, SYSTEM_A0, [])
            turn = conv.user(f"{ctx}\n\nQuestion: {q['question']}")
            steps = 1
        else:
            system = SYSTEM_A1 + (WIKI_HINT if CONFIGS[config]["root"] == "wiki" else "")
            conv = Conversation(model, system, kb.tools())
            turn = conv.user(q["question"])
            steps = 1
            while turn.stop == "tool_use" and turn.tool_calls:
                results = [(tc, kb.call(tc.name, tc.args)) for tc in turn.tool_calls]
                if steps >= MAX_STEPS:
                    results = [(tc, out + "\n\n[Tool-use limit reached. Do not call more tools; answer now from what "
                                           "you have found.]") for tc, out in results]
                    turn = conv.tool_results(results)
                    steps += 1
                    break
                turn = conv.tool_results(results)
                steps += 1
        rec.update({"answer": turn.text, "stop": turn.stop, "steps": steps})
    except Exception as e:  # recorded, never crashes the run
        rec.update({"answer": "", "stop": "error", "error": f"{type(e).__name__}: {e}"[:500],
                    "steps": locals().get("steps", 0)})
        conv = locals().get("conv")
    u = conv.usage if conv else None
    rec.update({
        "latency": round(time.time() - t0, 2),
        "retrieved": kb.ev.retrieved, "read": kb.ev.read, "read_raw": kb.ev.read_raw, "tool_calls": kb.ev.tool_calls,
        "tokens": {"input": u.input_tokens, "output": u.output_tokens, "cache_read": u.cache_read,
                   "cache_write": u.cache_write} if u else {},
        "model_cost": round(u.cost, 6) if u else 0.0,
        "retrieval_cost": round(kb.ev.retrieval_cost, 6),
    })
    rec["cost"] = round(rec["model_cost"] + rec["retrieval_cost"], 6)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["haiku", "sonnet", "deepseek", "qwen"])
    ap.add_argument("--configs", required=True, help="comma list of MODE:CONFIG, e.g. A0:R2_bm25,A1:R1_grep")
    ap.add_argument("--scales", default="300,900,3000")
    ap.add_argument("--sample", type=int, default=None, help="questions per scale (stratified); default all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, required=True, help="USD cap for this run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    spent = 0.0
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            done.add((r["qid"], r["config"], r["mode"], r["scale"], r["model"]))
            spent += r.get("cost", 0)
    jobs = []
    for scale in map(int, a.scales.split(",")):
        qs = load_questions(scale, a.sample)
        for mc in a.configs.split(","):
            mode, config = mc.split(":")
            if mode == "A0" and "search" not in CONFIGS[config]["tools"]:
                raise SystemExit(f"{config} has no retriever; A0 only works with R2/R3/R4")
            for q in qs:
                if (q["qid"], config, mode, scale, a.model) not in done:
                    jobs.append((a.model, mode, config, scale, q))
    print(f"{len(jobs)} jobs ({len(done)} already done, ${spent:.2f} spent so far); budget ${a.budget:.2f}", flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    n_done = 0

    def guarded(job):
        if stop.is_set():
            return None
        return run_one(*job)

    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(guarded, j) for j in jobs]
        for f in as_completed(futures):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                spent += rec["cost"]
                n_done += 1
                if n_done % 25 == 0 or rec["stop"] == "error":
                    print(f"  {n_done}/{len(jobs)}  ${spent:.2f}  last={rec['config']}/{rec['mode']} "
                          f"S{rec['scale']} {rec['stop']} {rec.get('error', '')[:120]}", flush=True)
                if spent >= a.budget and not stop.is_set():
                    stop.set()
                    print(f"budget ${a.budget:.2f} reached — stopping", flush=True)
    print(f"done: {n_done} records, ${spent:.2f} total", flush=True)


if __name__ == "__main__":
    main()
