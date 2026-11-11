"""Per-scale validity of BRIGHT unanswerable questions, with the same criterion the grader uses.

BRIGHT questions are open-ended forum questions. Removing their gold documents does not guarantee the knowledge
base lacks the topic: other pages may discuss it, so an assistant can give a substantive grounded answer (which the
grader then counts as "answered"). For each scale the question is asked at, candidates are retrieved from that
scale's corpus (BM25 top-6 + dense top-6) and Claude Sonnet 5.5 judges whether they allow a substantive, on-topic
answer. kb2_finalize.py keeps the question only at scales where they don't.
Output: questions_v2/validation_bright_scales.jsonl (per qid x scale, append-as-you-go).
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrievers.bm25 import BM25Index  # noqa: E402
from retrievers.corpus import corpus_dir, split_frontmatter  # noqa: E402
from retrievers.dense import DenseIndex  # noqa: E402

MODEL, PRICE = "claude-sonnet-5-5", (2.00, 10.00)
PER, DOC_CHARS = 6, 12000
OUT = ROOT / "questions_v2/validation_bright_scales.jsonl"

SYSTEM = """You audit a knowledge-base QA benchmark. A forum question is supposed to be unanswerable from the \
knowledge base. You see the question and the most relevant documents the knowledge base contains. Decide whether \
a careful assistant that may only state facts found in these documents could still give a substantive, on-topic \
answer to the question (directly addressing what is asked, not just a related fact or a loosely related topic)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "substantive_answer_possible": {"type": "boolean"},
        "supporting_docs": {"type": "array", "items": {"type": "integer"}},
        "evidence": {"type": "string"},
    },
    "required": ["substantive_answer_possible", "supporting_docs", "evidence"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)
lock = threading.Lock()
_idx: dict = {}


def indexes(scale: int):
    with lock:
        if scale not in _idx:
            _idx[scale] = (BM25Index(f"kb2/S{scale}"), DenseIndex(f"kb2/S{scale}"))
        return _idx[scale]


def check(q: dict, scale: int) -> tuple[dict, float]:
    bm, de = indexes(scale)
    paths: list[str] = []
    for hits in (bm.search(q["question_en"], 60), de.search(q["question_en"], 60)):
        n = 0
        for c, _ in hits:
            if c.path not in paths:
                paths.append(c.path)
                n += 1
            if n == PER:
                break
    raw = corpus_dir(f"kb2/S{scale}") / "raw"
    docs = "\n\n".join(f'<document index="{i}" path="{p}">\n{split_frontmatter((raw / p).read_text())[1][:DOC_CHARS]}'
                       f"\n</document>" for i, p in enumerate(paths))
    r = client.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                               output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                               messages=[{"role": "user", "content": f"{docs}\n\nQuestion:\n{q['question_en']}"}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']} S{scale}: stop_reason={r.stop_reason}")
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    v["supporting_paths"] = [paths[i] for i in v.pop("supporting_docs") if 0 <= i < len(paths)]
    return v, (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6


if __name__ == "__main__":
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl")) if q["category"] == "bright_unans"]
    done = {(r["qid"], r["scale"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    jobs = [(q, s) for q in qs for s in q["scales"] if (q["qid"], s) not in done]

    def run(job) -> float:
        q, s = job
        v, c = check(q, s)
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": q["qid"], "scale": s, **v}, ensure_ascii=False) + "\n")
        return c

    with ThreadPoolExecutor(12) as pool:
        cost = sum(pool.map(run, jobs))
    res = [json.loads(l) for l in OUT.read_text().splitlines()]
    from collections import Counter
    print(f"checked {len(jobs)} (cached {len(done)}), cost ${cost:.2f}")
    print("substantive answer possible, by scale:", dict(Counter((r["scale"], r["substantive_answer_possible"]) for r in res)))
