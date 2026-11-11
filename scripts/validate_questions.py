"""Corpus-level validity check for the main question set (RepliQA was written per-document, we use it per-corpus).

For each question, retrieve candidate docs from S3000 (superset of all scales) with BM25 + dense, drop the gold
doc, and ask Claude Sonnet 5.5 whether any candidate answers the question.
  - unanswerable question + some doc answers it        -> excluded (leak)
  - answerable question + some doc gives a different answer -> excluded (ambiguous)
  - answerable question + some doc gives the same answer    -> doc added to alt_gold_paths
Results are cached per qid in the `validation` field; excluded qids go to questions/excluded.json.
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
sys.path.insert(0, str(ROOT / "scripts"))

from retrievers.bm25 import BM25Index  # noqa: E402
from retrievers.corpus import corpus_dir, split_frontmatter  # noqa: E402
from retrievers.dense import DenseIndex  # noqa: E402

MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.00, 10.00
CORPUS = "repliqa/S3000"
PER_RETRIEVER, MAX_CANDIDATES = 6, 8
QFILE = ROOT / "questions/main.jsonl"
EXCLUDED_PATH = ROOT / "questions/excluded.json"

SYSTEM = """You audit a question-answering benchmark. You get a question, possibly a reference answer, and several \
candidate documents from a knowledge base. For each document decide whether it contains information that \
answers the question as asked — about the same specific people, organizations, places or events named in the \
question. A document about a similar topic but different named entities does not answer it. Generic questions \
(no named entities) are answered by any document that directly addresses them.
When a reference answer is given, also say whether the document's answer is consistent with it."""

SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc": {"type": "integer"},
                    "answers_question": {"type": "boolean"},
                    "consistent_with_reference": {"type": ["boolean", "null"]},
                    "evidence": {"type": "string"},
                },
                "required": ["doc", "answers_question", "consistent_with_reference", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["documents"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)


def candidates(q: dict, bm25: BM25Index, dense: DenseIndex) -> list[str]:
    exclude = set(q["gold_paths"])
    out: list[str] = []
    for hits in (bm25.search(q["question_en"], k=40), dense.search(q["question_en"], k=40)):
        n = 0
        for chunk, _ in hits:
            if chunk.path not in exclude and chunk.path not in out:
                out.append(chunk.path)
                n += 1
            if n == PER_RETRIEVER:
                break
    return out[:MAX_CANDIDATES]


def judge(q: dict, paths: list[str]) -> tuple[dict, float]:
    raw = corpus_dir(CORPUS) / "raw"
    docs = "\n\n".join(
        f'<document index="{i}">\n{split_frontmatter((raw / p).read_text())[1]}\n</document>' for i, p in enumerate(paths)
    )
    ref = f"\nReference answer: {q['answer']}" if q["answerable"] else "\n(No reference answer.)"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"{docs}\n\nQuestion: {q['question_en']}{ref}"}],
    )
    cost = (resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT) / 1e6
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']}: stop_reason={resp.stop_reason}")
    verdicts = json.loads(next(b.text for b in resp.content if b.type == "text"))["documents"]
    answering = [dict(v, path=paths[v["doc"]]) for v in verdicts if v["answers_question"] and 0 <= v["doc"] < len(paths)]
    if not q["answerable"]:
        status = "leak" if answering else "ok"
    elif any(v["consistent_with_reference"] is False for v in answering):
        status = "ambiguous"
    else:
        status = "ok"
    return {"status": status, "checked": paths, "answering": answering}, cost


if __name__ == "__main__":
    from build_questions import VALIDATION_CACHE, append_cache, apply_validation, load_cache

    items = [json.loads(l) for l in QFILE.read_text().splitlines()]
    cache = load_cache(VALIDATION_CACHE)
    todo = [q for q in items if q["qid"] not in cache]
    print(f"{len(todo)} questions to validate ({len(items) - len(todo)} cached)", flush=True)
    if todo:
        bm25, dense = BM25Index(CORPUS), DenseIndex(CORPUS)
        lock = threading.Lock()

        def run(q: dict) -> float:
            v, c = judge(q, candidates(q, bm25, dense))
            with lock:  # persist each result immediately
                append_cache(VALIDATION_CACHE, {"qid": q["qid"], "validation": v})
                cache[q["qid"]] = {"validation": v}
            return c

        with ThreadPoolExecutor(max_workers=12) as pool:
            cost = sum(pool.map(run, todo))
        print(f"validation cost ${cost:.2f}")
    for q in items:
        apply_validation(q, cache[q["qid"]]["validation"])
    QFILE.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in items), encoding="utf-8")

    excluded = set(json.loads(EXCLUDED_PATH.read_text())) if EXCLUDED_PATH.exists() else set()
    new = {q["qid"] for q in items if q.get("validation", {}).get("status") in ("leak", "ambiguous")}
    EXCLUDED_PATH.write_text(json.dumps(sorted(excluded | new), indent=0))
    from collections import Counter
    print("status by category:", dict(Counter((q["category"], q["validation"]["status"]) for q in items if "validation" in q)))
    print(f"newly excluded: {len(new - excluded)} | total excluded: {len(excluded | new)}")
    print("with alternative gold docs:", sum(1 for q in items if q.get("alt_gold_paths")))
