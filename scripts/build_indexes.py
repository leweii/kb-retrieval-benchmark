"""Build BM25 + dense indexes for the raw corpora.

Usage: build_indexes.py [corpus ...]   (default: all raw corpora)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrievers import bm25, dense  # noqa: E402

CORPORA = ["repliqa/S3000", "repliqa/S900", "repliqa/S300", "multihop",  # S3000 first: S900/S300 reuse its vectors
           "repliqa/S300@wiki", "repliqa/S900@wiki", "repliqa/S3000@wiki", "multihop@wiki"]
REUSE = {"repliqa/S900": "repliqa/S3000", "repliqa/S300": "repliqa/S3000"}

if __name__ == "__main__":
    total = 0.0
    for corpus in sys.argv[1:] or CORPORA:
        t = time.time()
        n = bm25.build(corpus)
        print(f"{corpus}: bm25 {n} chunks ({time.time() - t:.1f}s)", flush=True)
        t = time.time()
        n, cost = dense.build(corpus, reuse_from=REUSE.get(corpus))
        total += cost
        print(f"{corpus}: dense {n} chunks, ${cost:.4f} ({time.time() - t:.1f}s)")
    print(f"embedding cost total ${total:.4f}")
