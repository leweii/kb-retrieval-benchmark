"""R3 hybrid (BM25 + dense, RRF) and R4 enhanced hybrid (QMD-style pipeline, all via API).

R4 follows QMD's `qmd query` design with every local model replaced by an API call:
  1. query expansion (Claude Haiku 5.5): lex variants -> BM25, vec variants + HyDE passage -> dense
  2. RRF over all lists (original-query lists weighted x2) + top-rank bonus (+0.05 for #1, +0.02 for #2-3)
  3. rerank the top RERANK_CANDIDATES chunks with qwen3-reranker-8b (OpenRouter /rerank)
  4. position-aware blend of RRF and reranker scores (ranks 1-3: 75/25, 4-10: 60/40, 11+: 40/60)
Each search returns hits plus a cost/latency breakdown so the harness can account for retrieval overhead.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import anthropic
import httpx

from retrievers.bm25 import BM25Index
from retrievers.corpus import Chunk
from retrievers.dense import QUERY_INSTRUCTIONS, DenseIndex, embed

RRF_K = 60
CANDIDATES = 40  # per list
RERANK_CANDIDATES = 30
EXPANSION_MODEL = "claude-haiku-5-5"
EXPANSION_PRICE = (0.10, 0.50)
RERANK_MODEL = "qwen/qwen3-reranker-8b"
RERANK_PROVIDER = "Fireworks"
# P2 uses the Haiku LLM reranker (OpenRouter credits ran out; ~10x cheaper than qwen3-reranker at $0.20/M tokens).
RERANKER = "haiku"  # "haiku" | "api" (OpenRouter /rerank with RERANK_MODEL)
RERANK_PASSAGE_CHARS = 1500
RERANK_SYSTEM = """You rerank search results. For each passage, rate how useful it is for answering the query \
(the query may be in Chinese; passages are in English): 10 = directly contains the answer or the key fact needed, \
7 = clearly relevant and helpful, 4 = same topic but not helpful for this query, 0 = unrelated. Rate every \
passage independently, ignoring its position in the list."""
RERANK_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "score": {"type": "integer"}},
        "required": ["id", "score"], "additionalProperties": False}}},
    "required": ["scores"],
    "additionalProperties": False,
}


@dataclass
class SearchResult:
    hits: list[tuple[Chunk, float]]
    cost: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def rrf(lists: list[tuple[list[Chunk], float]], bonus: bool = False) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}
    for chunks, weight in lists:
        for rank, c in enumerate(chunks):
            by_id[c.chunk_id] = c
            scores[c.chunk_id] += weight / (RRF_K + rank + 1)
            if bonus:
                scores[c.chunk_id] += 0.05 if rank == 0 else 0.02 if rank < 3 else 0.0
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class Hybrid:
    """R3: BM25 + dense, plain RRF."""

    def __init__(self, corpus: str, bm25: BM25Index | None = None, dense: DenseIndex | None = None):
        self.bm25 = bm25 or BM25Index(corpus)  # pass shared, already-loaded indexes to avoid reloading
        self.dense = dense or DenseIndex(corpus)

    def search(self, query: str, k: int = 20) -> SearchResult:
        t = time.time()
        bm = [c for c, _ in self.bm25.search(query, CANDIDATES)]
        t_bm = time.time()
        qv, cost = embed([QUERY_INSTRUCTIONS.get(self.dense.model, "") + query], self.dense.model)
        de = [c for c, _ in self.dense.search(query, CANDIDATES, qvec=qv[0])]
        t_de = time.time()
        return SearchResult(rrf([(bm, 1.0), (de, 1.0)])[:k], cost,
                            {"bm25": t_bm - t, "dense": t_de - t_bm, "total": t_de - t})


EXPANSION_SYSTEM = """You expand a user's question for a hybrid search engine over a markdown knowledge base \
(English documents; the question may be in Chinese or English).
Return:
- lex: 2 short keyword queries in English for BM25 (names, distinctive terms, synonyms)
- vec: 2 natural-language English rephrasings of the question for semantic search
- hyde: a 2-3 sentence English passage that a document answering the question might contain
Keep all names of people, organizations and places exactly as given."""

EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "lex": {"type": "array", "items": {"type": "string"}},
        "vec": {"type": "array", "items": {"type": "string"}},
        "hyde": {"type": "string"},
    },
    "required": ["lex", "vec", "hyde"],
    "additionalProperties": False,
}


class EnhancedHybrid(Hybrid):
    """R4: QMD-style expansion + RRF + rerank + position-aware blend. Flags allow P1 ablations."""

    def __init__(self, corpus: str, expand: bool = True, rerank: bool = True,
                 bm25: BM25Index | None = None, dense: DenseIndex | None = None, reranker: str | None = None):
        super().__init__(corpus, bm25, dense)
        self.expand, self.rerank_on = expand, rerank
        self.reranker = reranker or RERANKER
        self.claude = anthropic.Anthropic(max_retries=5)
        self.http = httpx.Client(timeout=60)

    def rerank(self, query: str, chunks: list[Chunk]) -> tuple[list[float], float]:
        return self.rerank_haiku(query, chunks) if self.reranker == "haiku" else self.rerank_api(query, chunks)

    def rerank_haiku(self, query: str, chunks: list[Chunk]) -> tuple[list[float], float]:
        """LLM reranker (QMD's reranker is an LLM judging relevance): Claude Haiku 5.5 scores every candidate
        0-10 in one structured call; passages are cut to RERANK_PASSAGE_CHARS. Scores are scaled to 0-1."""
        passages = "\n\n".join(f'<passage id="{i}">\n{c.text[:RERANK_PASSAGE_CHARS]}\n</passage>'
                               for i, c in enumerate(chunks))
        resp = self.claude.messages.create(
            model=EXPANSION_MODEL,
            max_tokens=4000,
            system=RERANK_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": RERANK_SCHEMA}},
            messages=[{"role": "user", "content": f"{passages}\n\nQuery: {query}"}],
        )
        cost = (resp.usage.input_tokens * EXPANSION_PRICE[0] + resp.usage.output_tokens * EXPANSION_PRICE[1]) / 1e6
        scores = [0.0] * len(chunks)
        for item in json.loads(next(b.text for b in resp.content if b.type == "text"))["scores"]:
            if 0 <= item["id"] < len(chunks):
                scores[item["id"]] = max(0.0, min(10.0, float(item["score"]))) / 10
        return scores, cost

    def expansion(self, query: str) -> tuple[dict, float]:
        resp = self.claude.messages.create(
            model=EXPANSION_MODEL,
            max_tokens=4000,
            system=EXPANSION_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": EXPANSION_SCHEMA}},
            messages=[{"role": "user", "content": query}],
        )
        cost = (resp.usage.input_tokens * EXPANSION_PRICE[0] + resp.usage.output_tokens * EXPANSION_PRICE[1]) / 1e6
        return json.loads(next(b.text for b in resp.content if b.type == "text")), cost

    def rerank_api(self, query: str, chunks: list[Chunk]) -> tuple[list[float], float]:
        for attempt in range(6):
            r = self.http.post(
                "https://openrouter.ai/api/v1/rerank",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": RERANK_MODEL, "query": query, "documents": [c.text for c in chunks],
                      "provider": {"order": [RERANK_PROVIDER], "allow_fallbacks": False}},
            )
            if r.status_code == 200:
                break
            if attempt == 5:
                r.raise_for_status()
            time.sleep(2 ** (attempt + 1))
        data = r.json()
        scores = [0.0] * len(chunks)
        for item in data["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores, float(data.get("usage", {}).get("cost", 0) or 0)

    def search(self, query: str, k: int = 20) -> SearchResult:
        timings, cost, debug = {}, 0.0, {}
        t0 = time.time()
        lex_q, vec_q = [], []
        if self.expand:
            exp, c = self.expansion(query)
            cost += c
            lex_q, vec_q = exp["lex"], exp["vec"] + [exp["hyde"]]
            debug["expansion"] = exp
        timings["expansion"] = time.time() - t0

        t = time.time()
        instr = QUERY_INSTRUCTIONS.get(self.dense.model, "")
        texts = [instr + query] + [instr + v for v in vec_q[:-1]] + ([vec_q[-1]] if vec_q else [])  # HyDE: doc-side
        qvecs, c = embed(texts, self.dense.model)
        cost += c
        lists = [([x for x, _ in self.bm25.search(query, CANDIDATES)], 2.0),
                 ([x for x, _ in self.dense.search(query, CANDIDATES, qvec=qvecs[0])], 2.0)]
        lists += [([x for x, _ in self.bm25.search(q, CANDIDATES)], 1.0) for q in lex_q]
        lists += [([x for x, _ in self.dense.search("", CANDIDATES, qvec=v)], 1.0) for v in qvecs[1:]]
        fused = rrf(lists, bonus=True)
        timings["retrieve"] = time.time() - t

        if not self.rerank_on:
            timings["total"] = time.time() - t0
            return SearchResult(fused[:k], cost, timings, debug)

        t = time.time()
        cands = fused[:RERANK_CANDIDATES]
        rr, c = self.rerank(query, [x for x, _ in cands])
        cost += c
        top = cands[0][1]
        blended = []
        for rank, ((chunk, s), r) in enumerate(zip(cands, rr)):
            w = 0.75 if rank < 3 else 0.60 if rank < 10 else 0.40
            blended.append((chunk, w * (s / top) + (1 - w) * r))
        blended.sort(key=lambda x: -x[1])
        timings["rerank"] = time.time() - t
        timings["total"] = time.time() - t0
        return SearchResult(blended[:k], cost, timings, debug)
