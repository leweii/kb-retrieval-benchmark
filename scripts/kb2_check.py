"""Discriminative check on the built v2 knowledge base with the real question set (English versions).

Per category x scale: Recall@10 for BM25, dense and hybrid (RRF), counted at document level; for multi-doc
questions also 'all gold docs in top 10'. Unanswerable questions are skipped (no gold). No LLM calls; one
embedding call per question (shared across scales). Output: report/kb2_check.json + printed table.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrievers.bm25 import BM25Index  # noqa: E402
from retrievers.dense import QUERY_INSTRUCTIONS, DenseIndex, embed  # noqa: E402
from retrievers.hybrid import rrf  # noqa: E402

K, CAND = 10, 100


def doc_rank(hits) -> list[str]:
    out = []
    for c, _ in hits:
        if c.path not in out:
            out.append(c.path)
    return out


if __name__ == "__main__":
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl")) if q["answerable"]]
    qv, cost = embed([QUERY_INSTRUCTIONS["qwen/qwen3-embedding-8b"] + q["question_en"] for q in qs])
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # (cat, scale) -> method -> [any, all, multi, n]
    for scale in (300, 900, 3000):
        corpus = f"kb2/S{scale}"
        bm, de = BM25Index(corpus), DenseIndex(corpus)
        for q, v in zip(qs, qv):
            if scale not in q["scales"]:
                continue
            bmh = bm.search(q["question_en"], CAND)
            deh = de.search("", CAND, qvec=v)
            hy = rrf([([c for c, _ in bmh], 1.0), ([c for c, _ in deh], 1.0)])
            gold = set(q["gold_paths"])
            for name, hits in (("bm25", bmh), ("dense", deh), ("hybrid", hy)):
                top = set(doc_rank(hits)[:K])
                s = stats[(q["category"], scale)][name]
                s[0] += bool(top & gold)
                s[1] += gold <= top
                s[2] += len(gold) > 1
                s[3] += 1
    report = []
    print(f"{'category':12s} {'scale':>5s} {'n':>4s} | " + " | ".join(f"{m:>17s}" for m in ("bm25", "dense", "hybrid")))
    for (cat, scale), ms in sorted(stats.items()):
        row = {"category": cat, "scale": scale}
        cells = []
        for m in ("bm25", "dense", "hybrid"):
            a, al, multi, n = ms[m]
            row[m] = {"any": round(a / n, 2), "all": round(al / n, 2), "n": n}
            cells.append(f"any {a/n:4.0%} all {al/n:4.0%}" if multi else f"any {a/n:4.0%}         ")
        report.append(row)
        print(f"{cat:12s} {scale:5d} {ms['bm25'][3]:4d} | " + " | ".join(cells))
    (ROOT / "report/kb2_check.json").write_text(json.dumps(report, indent=1))
    print(f"embedding cost ${cost:.4f}")
