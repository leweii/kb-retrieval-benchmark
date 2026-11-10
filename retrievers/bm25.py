"""BM25 over chunks (bm25s, English stemming + stopwords). Pure local, no service."""

from __future__ import annotations

from pathlib import Path

import bm25s
import Stemmer

from retrievers.corpus import ROOT, Chunk, load_chunks, read_chunks, save_chunks

STEMMER = Stemmer.Stemmer("english")


def index_dir(corpus: str) -> Path:
    return ROOT / "indexes" / corpus / "bm25"


def build(corpus: str) -> int:
    chunks = load_chunks(corpus)
    tokens = bm25s.tokenize([c.text for c in chunks], stopwords="en", stemmer=STEMMER, show_progress=False)
    model = bm25s.BM25()
    model.index(tokens, show_progress=False)
    d = index_dir(corpus)
    model.save(str(d))
    save_chunks(chunks, d / "chunks.jsonl")
    return len(chunks)


class BM25Index:
    def __init__(self, corpus: str):
        d = index_dir(corpus)
        self.model = bm25s.BM25.load(str(d))
        self.chunks: list[Chunk] = read_chunks(d / "chunks.jsonl")

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        q = bm25s.tokenize([query], stopwords="en", stemmer=STEMMER, show_progress=False)
        if not q.ids[0]:  # e.g. query with no indexable tokens
            return []
        idx, scores = self.model.retrieve(q, k=min(k, len(self.chunks)), show_progress=False)
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0]) if s > 0]
