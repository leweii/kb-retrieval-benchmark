"""Dense retrieval: API embeddings (OpenRouter) stored as a local .npy matrix, brute-force cosine in numpy.

No vector database: ~10k chunks x 4096 dims is ~80MB in float16 and a full scan takes milliseconds.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from retrievers.corpus import ROOT, Chunk, approx_tokens, load_chunks, read_chunks, save_chunks

DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
# Qwen3-Embedding is instruction-aware on the query side only.
QUERY_INSTRUCTIONS = {
    "qwen/qwen3-embedding-8b": "Instruct: Given a question, retrieve passages from a knowledge base that answer it\nQuery: ",
}
# Pin one provider per model so corpus and query vectors come from the same deployment
# (providers differ slightly: mean cosine 0.997 between DeepInfra and SiliconFlow on the same texts).
PROVIDERS = {"qwen/qwen3-embedding-8b": "DeepInfra"}
BATCH_TOKENS = 16000  # per request; large batches trip provider per-minute token quotas
BATCH_MAX_ITEMS = 64

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_retries=5)
    return _client


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> tuple[np.ndarray, float]:
    """Returns (L2-normalized float32 matrix, cost in USD as reported by OpenRouter)."""
    batches: list[list[str]] = [[]]
    budget = 0
    for t in texts:
        n = approx_tokens(t)
        if batches[-1] and (budget + n > BATCH_TOKENS or len(batches[-1]) >= BATCH_MAX_ITEMS):
            batches.append([])
            budget = 0
        batches[-1].append(t)
        budget += n
    extra = {"provider": {"order": [PROVIDERS[model]], "allow_fallbacks": False}} if model in PROVIDERS else {}

    def one(batch: list[str]):
        attempts = 10
        for attempt in range(attempts):
            try:
                resp = client().embeddings.create(model=model, input=batch, encoding_format="float", extra_body=extra)
                vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
                if len(vecs) != len(batch):
                    raise RuntimeError(f"got {len(vecs)} embeddings for {len(batch)} inputs")
                cost = float(getattr(resp.usage, "cost", 0) or 0) if resp.usage else 0.0
                return vecs, cost
            except Exception:  # provider-side token quotas (429) recover within seconds
                if attempt == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, batches))
    mat = np.array([v for vecs, _ in results for v in vecs], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, sum(c for _, c in results)


def index_dir(corpus: str, model: str = DEFAULT_MODEL) -> Path:
    return ROOT / "indexes" / corpus / ("dense-" + model.replace("/", "_"))


def build(corpus: str, model: str = DEFAULT_MODEL, reuse_from: str | None = None) -> tuple[int, float]:
    """reuse_from: a superset corpus whose index already holds vectors for identical chunks (e.g. S3000 for
    S300/S900, which share files and chunking) — vectors are copied instead of re-embedded."""
    chunks = load_chunks(corpus)
    cost = 0.0
    if reuse_from:
        src = DenseIndex(reuse_from, model)
        pos = {c.chunk_id: i for i, c in enumerate(src.chunks)}
        for c in chunks:
            if c.chunk_id not in pos or src.chunks[pos[c.chunk_id]].text != c.text:
                raise ValueError(f"{c.chunk_id} not reusable from {reuse_from}")
        mat = src.vectors[[pos[c.chunk_id] for c in chunks]]
    else:
        mat, cost = embed([c.text for c in chunks], model)
    d = index_dir(corpus, model)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "vectors.npy", mat.astype(np.float16))
    save_chunks(chunks, d / "chunks.jsonl")
    return len(chunks), cost


class DenseIndex:
    def __init__(self, corpus: str, model: str = DEFAULT_MODEL):
        d = index_dir(corpus, model)
        self.model = model
        self.vectors = np.load(d / "vectors.npy").astype(np.float32)
        self.chunks: list[Chunk] = read_chunks(d / "chunks.jsonl")

    def embed_query(self, query: str) -> np.ndarray:
        vec, _ = embed([QUERY_INSTRUCTIONS.get(self.model, "") + query], self.model)
        return vec[0]

    def search(self, query: str, k: int = 20, qvec: np.ndarray | None = None) -> list[tuple[Chunk, float]]:
        q = self.embed_query(query) if qvec is None else qvec
        scores = self.vectors @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]
