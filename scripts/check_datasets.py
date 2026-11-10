"""Free discriminative check of candidate datasets (no model API calls).

For each dataset: sample up to N_QUERIES queries, build a ~CORPUS_SIZE-doc corpus from the same dataset (all gold
docs of the sampled queries + random same-dataset distractors), then report
  - overlap: median share of the query's content words that occur verbatim in its gold doc(s)
  - BM25 Recall@10 (any gold doc in top 10) and, for multi-doc questions, complete Recall@10 (all gold docs)
  - doc length (chars, median)
A dataset is a good retrieval benchmark if overlap is moderate and BM25 is clearly below ceiling.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

import bm25s
import jieba
import pandas as pd
import Stemmer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
N_QUERIES, CORPUS_SIZE, SEED = 100, 3000, 13
STEM = Stemmer.Stemmer("english")
STOP_EN = set(bm25s.stopwords.STOPWORDS_EN)
jieba.setLogLevel(60)


def en_tokens(text: str) -> list[str]:
    return [STEM.stemWord(w) for w in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()) if w not in STOP_EN and len(w) > 1]


ZH_STOP = set("的 了 是 在 和 有 我 你 他 她 它 们 这 那 吗 呢 吧 啊 怎么 什么 多少 哪 哪个 如何 为什么 一个 与 及 或 也 都 就 而 被 把 对 从 到 为 以 上 下 中 等".split())


def zh_tokens(text: str) -> list[str]:
    return [w for w in jieba.lcut_for_search(text.lower()) if w.strip() and w not in ZH_STOP and not re.fullmatch(r"\W+", w)]


# ---------------------------------------------------------------- loaders: -> (queries, pool)
# queries: [{"q": str, "gold": [doc_id, ...], "answerable": bool}] ; pool: {doc_id: text}

def load_musique():
    rows = [json.loads(l) for l in open(RAW / "musique/musique_full_v1.0_dev.jsonl")]
    pool, qs = {}, []
    for r in rows:
        gold = []
        for p in r["paragraphs"]:
            did = f"{p['title']}::{hash(p['paragraph_text']) & 0xffffffff:x}"
            pool[did] = f"{p['title']}\n{p['paragraph_text']}"
            if p["is_supporting"]:
                gold.append(did)
        qs.append({"q": r["question"], "gold": gold, "answerable": r["answerable"]})
    return qs, pool


def load_2wiki():
    rows = json.load(open(RAW / "2wiki/dev.json"))
    pool, qs = {}, []
    for r in rows:
        for title, sents in r["context"]:
            pool[title] = f"{title}\n" + " ".join(sents)
        qs.append({"q": r["question"], "gold": sorted({t for t, _ in r["supporting_facts"]}), "answerable": True})
    return qs, pool


def load_hotpot():
    df = pd.read_parquet(RAW / "hotpotqa/validation-00000-of-00001.parquet")
    pool, qs = {}, []
    for r in df.itertuples():
        for title, sents in zip(r.context["title"], r.context["sentences"]):
            pool[title] = f"{title}\n" + "".join(sents)
        qs.append({"q": r.question, "gold": sorted(set(r.supporting_facts["title"])), "answerable": True})
    return qs, pool


def load_nolima(rng: random.Random):
    """Book text split into ~500-word chunks; each needle instance inserted into its own random chunk."""
    words = []
    for f in sorted((RAW / "nolima/books").rglob("*.txt")):
        words += f.read_text(errors="ignore").split()
    chunks = [" ".join(words[i : i + 500]) for i in range(0, len(words), 500)]
    pool = {f"chunk{i:05d}": c for i, c in enumerate(chunks)}
    qs, free = [], list(pool)
    rng.shuffle(free)
    for needle in json.load(open(RAW / "nolima/needle_set.json")):
        for t in needle["tests"].values():
            args = t["input_args"]
            char = rng.choice(needle["character_set"])
            text = needle["needle"].replace("{CHAR}", char)
            for i, a in enumerate(args, 1):
                text = text.replace("{" + str(i) + "}", a)
            did = free.pop()
            sents = pool[did].split(". ")
            k = rng.randrange(len(sents))
            pool[did] = ". ".join(sents[:k] + [text] + sents[k:])
            for hop, qt in needle["questions"].items():
                q = qt
                for i, a in enumerate(args, 1):
                    q = q.replace("{" + str(i) + "}", a)
                qs.append({"q": q, "gold": [did], "answerable": True, "hop": hop})
    return qs, pool


def load_bright():
    pool, qs = {}, []
    for f in sorted((RAW / "bright/examples").glob("*.parquet")):
        dom = f.stem
        docs = pd.read_parquet(RAW / "bright/long_documents" / f.name)
        for r in docs.itertuples():
            pool[f"{dom}/{r.id}"] = r.content
        for r in pd.read_parquet(f).itertuples():
            gold = [f"{dom}/{g}" for g in r.gold_ids_long if f"{dom}/{g}" in pool]
            if gold:
                qs.append({"q": r.query, "gold": gold, "answerable": True})
    return qs, pool


def load_cmteb(name: str):
    base = RAW / name
    corpus = pd.read_parquet(base / "corpus/dev-00000-of-00001.parquet")
    queries = pd.read_parquet(base / "queries/dev-00000-of-00001.parquet")
    qrels = pd.read_parquet(base / "data/dev-00000-of-00001.parquet")
    qrels = qrels[qrels.score > 0]
    pool = dict(zip(corpus["_id"].astype(str), (corpus.title.fillna("") + "\n" + corpus.text).str.strip()))
    gold = qrels.groupby(qrels["query-id"].astype(str))["corpus-id"].apply(lambda s: [str(x) for x in s]).to_dict()
    qs = [{"q": t, "gold": gold[i], "answerable": True} for i, t in zip(queries["_id"].astype(str), queries.text) if i in gold]
    return qs, pool


def load_scifact():
    pool = {}
    for l in open(RAW / "scifact/corpus.jsonl"):
        d = json.loads(l)
        pool[str(d["_id"])] = f"{d.get('title', '')}\n{d['text']}"
    queries = {str(q["_id"]): q["text"] for q in map(json.loads, open(RAW / "scifact/queries.jsonl"))}
    gold: dict[str, list[str]] = {}
    for l in open(RAW / "scifact/qrels/test.jsonl"):
        r = json.loads(l)
        gold.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))
    qs = [{"q": queries[i], "gold": g, "answerable": True} for i, g in gold.items() if i in queries]
    return qs, pool


# ---------------------------------------------------------------- check

def check(name: str, qs: list[dict], pool: dict[str, str], zh: bool, rng: random.Random) -> dict:
    qs = [q for q in qs if q["answerable"] and q["gold"] and all(g in pool for g in q["gold"])]
    sample = rng.sample(qs, min(N_QUERIES, len(qs)))
    gold_ids = {g for q in sample for g in q["gold"]}
    rest = [d for d in pool if d not in gold_ids]
    corpus_ids = list(gold_ids) + rng.sample(rest, max(0, min(len(rest), CORPUS_SIZE - len(gold_ids))))
    tok = zh_tokens if zh else en_tokens
    corpus_tokens = [tok(pool[d]) for d in corpus_ids]
    model = bm25s.BM25()
    model.index(corpus_tokens, show_progress=False)
    doc_sets = {d: set(t) for d, t in zip(corpus_ids, corpus_tokens)}

    overlap, any10, all10, multi = [], 0, 0, 0
    for q in sample:
        qt = set(tok(q["q"]))
        gold_words = set().union(*(doc_sets[g] for g in q["gold"]))
        overlap.append(len(qt & gold_words) / max(1, len(qt)))
        res, _ = model.retrieve([list(qt) or ["_"]], k=10, show_progress=False)
        top = {corpus_ids[i] for i in res[0]}
        any10 += bool(top & set(q["gold"]))
        if len(q["gold"]) > 1:
            multi += 1
            all10 += set(q["gold"]) <= top
    out = {
        "dataset": name, "queries": len(sample), "corpus": len(corpus_ids),
        "doc_chars_p50": int(statistics.median(len(pool[d]) for d in corpus_ids)),
        "gold_per_q": round(statistics.mean(len(q["gold"]) for q in sample), 1),
        "overlap_p50": round(statistics.median(overlap), 2),
        "bm25_recall10_any": round(any10 / len(sample), 2),
        "bm25_recall10_all": round(all10 / multi, 2) if multi else None,
    }
    if name in DENSE:
        out.update(dense_check(sample, corpus_ids, pool))
    return out


DENSE = {"MuSiQue", "2WikiMultihopQA", "NoLiMa", "BRIGHT(StackExchange)"}


def dense_check(sample: list[dict], corpus_ids: list[str], pool: dict[str, str]) -> dict:
    """Same sample/corpus, dense retrieval with the experiment's embedding setup (qwen3-embedding-8b via
    OpenRouter, pinned provider). Long docs are chunked; a doc scores as its best chunk."""
    import numpy as np

    from retrievers.corpus import chunk_doc
    from retrievers.dense import QUERY_INSTRUCTIONS, embed

    chunk_doc_ids, chunk_texts = [], []
    for d in corpus_ids:
        for c in chunk_doc(d, pool[d]):
            chunk_doc_ids.append(d)
            chunk_texts.append(c.text)
    cvec, cost1 = embed(chunk_texts)
    qvec, cost2 = embed([QUERY_INSTRUCTIONS["qwen/qwen3-embedding-8b"] + q["q"] for q in sample])
    pos = {d: i for i, d in enumerate(corpus_ids)}
    owner = np.array([pos[d] for d in chunk_doc_ids])
    any10 = all10 = multi = 0
    for q, v in zip(sample, qvec):
        s = cvec @ v
        best = np.full(len(corpus_ids), -9.0)
        np.maximum.at(best, owner, s)
        top = {corpus_ids[i] for i in np.argsort(-best)[:10]}
        any10 += bool(top & set(q["gold"]))
        if len(q["gold"]) > 1:
            multi += 1
            all10 += set(q["gold"]) <= top
    return {"dense_recall10_any": round(any10 / len(sample), 2),
            "dense_recall10_all": round(all10 / multi, 2) if multi else None,
            "dense_chunks": len(chunk_texts), "embed_cost": round(cost1 + cost2, 4)}


if __name__ == "__main__":
    rng = random.Random(SEED)
    loaders = {
        "MuSiQue": (load_musique, False), "2WikiMultihopQA": (load_2wiki, False), "HotpotQA": (load_hotpot, False),
        "NoLiMa": (lambda: load_nolima(rng), False), "BRIGHT(StackExchange)": (load_bright, False),
        "DuRetrieval": (lambda: load_cmteb("DuRetrieval"), True), "T2Retrieval": (lambda: load_cmteb("T2Retrieval"), True),
        "SciFact": (load_scifact, False),
    }
    only = set(sys.argv[1:])
    out = []
    for name, (fn, zh) in loaders.items():
        if only and name not in only:
            continue
        qs, pool = fn()
        r = check(name, qs, pool, zh, rng)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report/dataset_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
