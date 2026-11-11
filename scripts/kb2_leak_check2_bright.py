"""v2 leak check, second pass, BRIGHT: can a BRIGHT "unanswerable" question still be answered substantively?

The first check (kb2_validate_bright.py) showed the judge each candidate document cut to its first 12,000
characters. BRIGHT pages are long web pages (often navigation first), so relevant text further down was never
seen — e.g. a US Treasury page on the Russian oil price cap whose first mention of the cap is at character
28,901. This pass works on chunks (no truncation) and adds the documents that answering runs actually read:
  1. the question: BM25 top-10 + dense top-10 chunks (S3000)
  2. for the documents most read by runs that answered the question: their 3 best BM25 chunks
Claude Opus 5.5 applies the original criterion (a substantive, on-topic answer from these passages alone) and
lists every passage that suffices on its own. The question leaks at a scale when a sufficient passage's document
is in that scale's corpus (corpora are nested).

Results -> questions_v2/leak_check2_bright.jsonl (per qid, resumable). Applied by kb2_finalize.py.

  uv run python -I scripts/kb2_leak_check2_bright.py [--limit N]
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bm25s  # noqa: E402

from harness.build_report import Data  # noqa: E402
from harness.run import BASE_RULES  # noqa: E402
from retrievers.bm25 import BM25Index  # noqa: E402
from retrievers.corpus import corpus_dir  # noqa: E402
from retrievers.dense import DenseIndex  # noqa: E402
from scripts.kb2_leak_check2 import MODEL, PRICE, client  # noqa: E402

CORPUS = "kb2/S3000"
K_Q, READ_DOCS, PER_DOC = 10, 8, 3
OUT = ROOT / "questions_v2/leak_check2_bright.jsonl"

SYSTEM = """You audit a knowledge-base QA benchmark. A forum question is supposed to be unanswerable from the \
knowledge base (its source documents were removed). You see the question and passages that remain in the \
knowledge base. Decide whether a careful assistant that may only state facts found in these passages could \
still give a substantive, on-topic answer to the question — directly addressing what is asked, not just a \
related fact or a loosely related topic. The assistant follows these rules:
<rules>
{rules}
</rules>
List in sufficient_passages every passage that on its own allows such an answer.""".replace("{rules}", BASE_RULES)

SCHEMA = {
    "type": "object",
    "properties": {
        "substantive_answer_possible": {"type": "boolean"},
        "sufficient_passages": {"type": "array", "items": {"type": "integer"}},
        "evidence": {"type": "string"},
    },
    "required": ["substantive_answer_possible", "sufficient_passages", "evidence"],
    "additionalProperties": False,
}

lock = threading.Lock()


def best_chunks_in_doc(chunks: list, query: str, k: int) -> list:
    if len(chunks) <= k:
        return chunks
    m = bm25s.BM25()
    m.index(bm25s.tokenize([c.text for c in chunks], stopwords="en", show_progress=False), show_progress=False)
    idx, _ = m.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False), k=k, show_progress=False)
    return [chunks[i] for i in idx[0]]


def candidates(q: dict, read_docs: list[str], bm: BM25Index, de: DenseIndex, by_doc: dict) -> list:
    out, seen = [], set()
    for hits in (bm.search(q["question_en"], K_Q), de.search(q["question_en"], K_Q)):
        for c, _ in hits:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                out.append(c)
    for p in read_docs:
        for c in best_chunks_in_doc(by_doc.get(p, []), q["question_en"], PER_DOC):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                out.append(c)
    return out


def judge(q: dict, chunks: list) -> tuple[dict, float]:
    passages = "\n\n".join(f'<passage index="{i}" path="{c.path}">\n{c.text}\n</passage>' for i, c in enumerate(chunks))
    r = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"{passages}\n\nQuestion:\n{q['question_en']}"}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']}: stop_reason={r.stop_reason}")
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    v["sufficient_paths"] = sorted({chunks[i].path for i in v.pop("sufficient_passages") if 0 <= i < len(chunks)})
    v["checked"] = [c.chunk_id for c in chunks]
    return v, (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    d = Data()
    qs = [q for q in d.qs.values() if q["category"] == "bright_unans"]
    read = {q["qid"]: Counter() for q in qs}
    for k, r in d.judged.items():
        if k[0] in read and r["judge"]["abstained"] is False:
            read[k[0]].update(set(r.get("read") or []))
    done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [q for q in qs if q["qid"] not in done][:limit]
    bm, de = BM25Index(CORPUS), DenseIndex(CORPUS)
    by_doc: dict[str, list] = {}
    for c in bm.chunks:
        by_doc.setdefault(c.path, []).append(c)

    def run(q: dict) -> float:
        docs = [p for p, _ in read[q["qid"]].most_common(READ_DOCS)]
        v, c = judge(q, candidates(q, docs, bm, de, by_doc))
        v["leaked_scales"] = [s for s in q["scales"] if any((corpus_dir(f"kb2/S{s}") / "raw" / p).exists()
                                                             for p in v["sufficient_paths"])] \
            if v["substantive_answer_possible"] else []
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": q["qid"], "question_en": q["question_en"][:300], **v, "cost": c},
                                   ensure_ascii=False) + "\n")
        return c

    with ThreadPoolExecutor(8) as pool:
        cost = sum(pool.map(run, todo))
    res = [json.loads(l) for l in OUT.read_text().splitlines()]
    print(f"checked {len(todo)} (cached {len(done)}), cost ${cost:.2f}")
    print("substantive answer possible:", sum(r["substantive_answer_possible"] for r in res), "of", len(res))
    for r in res:
        if r["substantive_answer_possible"]:
            print(f"  {r['qid']} scales={r['leaked_scales']} <- {r['sufficient_paths']}")
