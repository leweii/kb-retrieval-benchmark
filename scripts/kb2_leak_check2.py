"""v2 leak check, second pass: can a MuSiQue "unanswerable" question still be answered from the knowledge base?

An unanswerable MuSiQue question had the document of its last hop removed. The first check (kb2_validate.py)
retrieved candidates with the full question only. The last hop is about the bridge entity — named nowhere in the
question — so that check could not find a page that restates the removed fact (e.g. "Vatican City ... the
foundation of the state in 1929" on another page). This pass builds candidate passages from
  1. the last-hop question with #k replaced by the earlier hops' answers (BM25 + dense, chunk level)
  2. the full question (BM25 + dense), as before
  3. every chunk containing the original answer, preferring chunks that also name the bridge entity
and asks Claude Opus 5.5 whether an assistant bound by the benchmark's answering rules (harness/run.py
BASE_RULES) could give a substantive answer. Hops before the last remain answerable from the knowledge base.

A question leaks at a scale when all its supporting documents are in that scale's corpus.
Results -> questions_v2/leak_check2.jsonl (per qid, resumable). Applied by kb2_finalize.py.

  uv run python -I scripts/kb2_leak_check2.py [--limit N]
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.run import BASE_RULES  # noqa: E402
from retrievers.bm25 import BM25Index  # noqa: E402
from retrievers.corpus import corpus_dir  # noqa: E402
from retrievers.dense import DenseIndex  # noqa: E402

MODEL, PRICE = "claude-opus-5-5", (4.00, 20.00)
CORPUS, SCALES = "kb2/S3000", (300, 900, 3000)
K_LAST, K_FULL, K_ANSWER, MAX_CHUNKS = 10, 6, 10, 40
MUSIQUE = ROOT / "data/raw/musique/musique_full_v1.0_dev.jsonl"
OUT = ROOT / "questions_v2/leak_check2.jsonl"

SYSTEM = """You audit a knowledge-base QA benchmark. A multi-hop question was made unanswerable by removing \
the document that supports its last hop. You must decide whether the knowledge base can still answer it.

The assistant being tested follows these rules:
<rules>
{rules}
</rules>

You see the question, its decomposition into hops (the answers of all hops before the last are available in \
the knowledge base), the original answer, and candidate passages that remain in the knowledge base. Decide \
whether an assistant following the rules could give a substantive answer to the question from these passages \
(together with the earlier hops) — the original answer, a less specific but still correct form of it (for \
example the year of a full date), or another answer the passages support. General knowledge may only \
interpret and connect what the passages say, as the rules allow. Similar topics, or passages that only cover \
the earlier hops, do not count.""".replace("{rules}", BASE_RULES)

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "supporting_passages": {"type": "array", "items": {"type": "integer"}},
        "evidence": {"type": "string"},
    },
    "required": ["answerable", "answer", "supporting_passages", "evidence"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=6)
lock = threading.Lock()


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", s.lower()).split())


def last_hop_query(row: dict) -> str:
    hops = row["question_decomposition"]
    q = hops[-1]["question"]
    for i, h in enumerate(hops[:-1], 1):
        q = q.replace(f"#{i}", h["answer"])
    return q


def candidates(q: dict, row: dict, bm: BM25Index, de: DenseIndex, norm_chunks: list[str]) -> list:
    out, seen = [], set()

    def add(chunks, n):
        added = 0
        for c in chunks:
            if c.chunk_id not in seen and len(out) < MAX_CHUNKS:
                seen.add(c.chunk_id)
                out.append(c)
                added += 1
            if added == n:
                break

    lq = last_hop_query(row)
    add([c for c, _ in bm.search(lq, 40)], K_LAST)
    add([c for c, _ in de.search(lq, 40)], K_LAST)
    add([c for c, _ in bm.search(q["question_en"], 40)], K_FULL)
    add([c for c, _ in de.search(q["question_en"], 40)], K_FULL)
    ans = norm(str(q.get("original_answer") or ""))
    bridge = norm(row["question_decomposition"][-2]["answer"]) if len(row["question_decomposition"]) > 1 else ""
    if len(ans) >= 4:
        hits = [c for c, t in zip(bm.chunks, norm_chunks) if f" {ans} " in f" {t} "]
        hits.sort(key=lambda c: (bridge not in norm_chunks_by_id[c.chunk_id], len(c.text)))
        add(hits, K_ANSWER)
    return out


def judge(q: dict, row: dict, chunks: list) -> tuple[dict, float]:
    passages = "\n\n".join(f'<passage index="{i}" path="{c.path}">\n{c.text}\n</passage>' for i, c in enumerate(chunks))
    hops = "\n".join(f"{i}. {h['question']}  ->  {h['answer']}"
                     for i, h in enumerate(row["question_decomposition"], 1))
    user = (f"{passages}\n\nQuestion: {q['question_en']}\n\nDecomposition (#k = answer of hop k; the last hop's "
            f"document was removed):\n{hops}\n\nOriginal answer: {q.get('original_answer')}")
    r = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']}: stop_reason={r.stop_reason}")
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    v["supporting_paths"] = sorted({chunks[i].path for i in v.pop("supporting_passages") if 0 <= i < len(chunks)})
    v["checked"] = [c.chunk_id for c in chunks]
    v["last_hop_query"] = last_hop_query(row)
    return v, (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6


def leaked_scales(v: dict, scales: list[int]) -> list[int]:
    if not v["answerable"] or not v["supporting_paths"]:
        return []
    return [s for s in scales if all((corpus_dir(f"kb2/S{s}") / "raw" / p).exists() for p in v["supporting_paths"])]


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl")) if q["category"] == "musique_unans"]
    want = {q["src_id"] for q in qs}
    rows = {r["id"]: r for r in map(json.loads, open(MUSIQUE)) if r["id"] in want}
    done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [q for q in qs if q["qid"] not in done][:limit]
    bm, de = BM25Index(CORPUS), DenseIndex(CORPUS)
    norm_chunks = [norm(c.text) for c in bm.chunks]
    norm_chunks_by_id = {c.chunk_id: t for c, t in zip(bm.chunks, norm_chunks)}

    def run(q: dict) -> float:
        row = rows[q["src_id"]]
        v, c = judge(q, row, candidates(q, row, bm, de, norm_chunks))
        v["leaked_scales"] = leaked_scales(v, q["scales"])
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": q["qid"], "question_en": q["question_en"],
                                    "original_answer": q.get("original_answer"), **v, "cost": c},
                                   ensure_ascii=False) + "\n")
        return c

    with ThreadPoolExecutor(8) as pool:
        cost = sum(pool.map(run, todo))
    res = [json.loads(l) for l in OUT.read_text().splitlines()]
    print(f"checked {len(todo)} (cached {len(done)}), cost ${cost:.2f}")
    print("answerable:", sum(r["answerable"] for r in res), "of", len(res))
    for r in res:
        if r["answerable"]:
            print(f"  {r['qid']} scales={r['leaked_scales']} {r['answer'][:60]!r} <- {r['supporting_paths']}")
