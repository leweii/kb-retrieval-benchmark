"""v2: leak check for unanswerable questions on kb2/S3000 (superset of all scales).

An unanswerable question had its evidence removed from every corpus. Wikipedia is redundant and BRIGHT pages
overlap, so other documents may still answer it. Candidates = BM25 top-6 + dense top-6 documents; Claude
Sonnet 5.5 judges whether any candidate answers the question (the original answer is shown as a reference).
Results -> questions_v2/validation.jsonl (per qid, append-as-you-go). Does not modify questions.jsonl.
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
CORPUS, PER, MAXC, DOC_CHARS = "kb2/S3000", 6, 10, 12000
OUT = ROOT / "questions_v2/validation.jsonl"

SYSTEM = """You audit a knowledge-base QA benchmark. A question was made unanswerable by removing its evidence \
from the knowledge base. You see the question, the answer it originally had, and candidate documents that \
remain in the knowledge base. Decide for each document whether it (alone or together with the other \
candidates) contains the information needed to answer the question with that answer. Similar topics or \
partial steps of a multi-step question do not count — only the missing information itself."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable_from_candidates": {"type": "boolean"},
        "supporting_docs": {"type": "array", "items": {"type": "integer"}},
        "evidence": {"type": "string"},
    },
    "required": ["answerable_from_candidates", "supporting_docs", "evidence"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)
lock = threading.Lock()


def candidates(q: dict, bm: BM25Index, de: DenseIndex) -> list[str]:
    out: list[str] = []
    for hits in (bm.search(q["question_en"], 60), de.search(q["question_en"], 60)):
        n = 0
        for c, _ in hits:
            if c.path not in out:
                out.append(c.path)
                n += 1
            if n == PER:
                break
    return out[:MAXC]


def judge(q: dict, paths: list[str]) -> tuple[dict, float]:
    raw = corpus_dir(CORPUS) / "raw"
    docs = "\n\n".join(f'<document index="{i}" path="{p}">\n{split_frontmatter((raw / p).read_text())[1][:DOC_CHARS]}'
                       f"\n</document>" for i, p in enumerate(paths))
    user = (f"{docs}\n\nQuestion: {q['question_en']}\n\nOriginal answer (its evidence was removed):\n"
            f"{str(q.get('original_answer'))[:4000]}")
    r = client.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                               output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                               messages=[{"role": "user", "content": user}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']}: stop_reason={r.stop_reason}")
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    v["supporting_paths"] = [paths[i] for i in v.pop("supporting_docs") if 0 <= i < len(paths)]
    v["checked"] = paths
    return v, (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6


if __name__ == "__main__":
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl")) if not q["answerable"]]
    done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    bm, de = BM25Index(CORPUS), DenseIndex(CORPUS)

    def run(q) -> float:
        v, c = judge(q, candidates(q, bm, de))
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": q["qid"], "category": q["category"], **v}, ensure_ascii=False) + "\n")
        return c

    with ThreadPoolExecutor(12) as pool:
        cost = sum(pool.map(run, todo))
    res = [json.loads(l) for l in OUT.read_text().splitlines()]
    from collections import Counter
    print(f"validated {len(todo)} (cached {len(done)}), cost ${cost:.2f}")
    print("leaks by category:", dict(Counter((r["category"], r["answerable_from_candidates"]) for r in res)))
