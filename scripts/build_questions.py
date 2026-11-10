"""Build question sets (English; Chinese translation is a separate step).

questions/main.jsonl   RepliQA — used at all scales (see `scales` field per question)
  answerable    200  gold doc in S300, one question per doc
  unans_absent_info  50  doc in S300 but the answer is not in it (RepliQA native "not found")
  unans_doc_missing  50  answerable question whose doc (from split 1) is in no corpus
  sanity_900     50  gold doc in S900 \\ S300   (asked at S900, S3000)
  sanity_3000    50  gold doc in S3000 \\ S900  (asked at S3000 only)
questions/side.jsonl   MultiHop-RAG — 150 questions, stratified by type

Half of every category is marked lang=zh for translation (deterministic by qid hash). Questions listed in
questions/excluded.json (written by validate_questions.py) are skipped and refilled from the pool.
Existing translations and validation results are carried over by qid.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEED = 7
NOT_FOUND = "The answer is not found in the document."

MAIN_PLAN = {  # category -> (count, source pool, scales asked at, answerable)
    "answerable": (200, "s300", [300, 900, 3000], True),
    "unans_absent_info": (50, "s300_rest", [300, 900, 3000], False),
    "unans_doc_missing": (50, "split1", [300, 900, 3000], False),
    "sanity_900": (50, "s900_only", [900, 3000], True),
    "sanity_3000": (50, "s3000_only", [3000], True),
}
SIDE_PLAN = {"inference_query": 40, "comparison_query": 40, "temporal_query": 35, "null_query": 35}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# RepliQA questions were written against a single document; phrasings like "according to the document"
# presuppose that and don't occur in real knowledge-base queries.
SINGLE_DOC_PHRASING = r"(?i)\b(?:the|this) (?:document|article|text|passage|piece)\b"
EXCLUDED_PATH = ROOT / "questions/excluded.json"
EXCLUDED: set[str] = set(json.loads(EXCLUDED_PATH.read_text())) if EXCLUDED_PATH.exists() else set()


def assign_lang(items: list[dict]) -> None:
    by_cat = defaultdict(list)
    for q in items:
        by_cat[q["category"]].append(q)
    for qs in by_cat.values():
        qs.sort(key=lambda q: hashlib.sha1(q["qid"].encode()).hexdigest())
        for i, q in enumerate(qs):
            q["lang"] = "zh" if i < len(qs) // 2 else "en"


def build_main(rng: random.Random) -> list[dict]:
    manifest = {d["doc_id"]: d for d in load_jsonl(ROOT / "corpora/repliqa/S3000/docs.jsonl")}
    q0 = pd.read_parquet(RAW / "repliqa/repliqa_0.parquet")
    q1 = pd.read_parquet(RAW / "repliqa/repliqa_1.parquet")
    q0 = q0[q0.document_id.isin(manifest)]
    q0 = q0.assign(min_scale=q0.document_id.map(lambda i: manifest[i]["min_scale"]),
                   answerable=q0.answer.str.strip() != NOT_FOUND)

    # Selection is stable under exclusions: doc order and per-doc question choice are hash-based, so excluding
    # a question only advances that doc to its next candidate instead of reshuffling everything downstream.
    def hashed(keys, salt: str) -> list[str]:
        return sorted(keys, key=lambda k: hashlib.sha1(f"{salt}:{k}".encode()).hexdigest())

    s300_docs = hashed(q0[q0.min_scale == 300].document_id.unique(), "s300")
    answerable_docs, rest_docs = s300_docs[:200], s300_docs[200:]

    validated_ok = {qid for qid, r in load_cache(VALIDATION_CACHE).items() if r["validation"]["status"] == "ok"}
    taken: set[str] = set()  # one question per doc per pool call is enforced below; across categories too

    def pick(df: pd.DataFrame, docs: list[str], n: int, answerable: bool) -> list[pd.Series]:
        """Prefer questions that already passed validation (saves re-validating), then hash order.
        Both are deterministic given the committed caches; no bias beyond the validity filter itself."""
        eligible = df[(df.answerable == answerable) & ~("m-" + df.question_id).isin(EXCLUDED)
                      & ~("m-" + df.question_id).isin(taken)
                      & ~df.question.str.contains(SINGLE_DOC_PHRASING)]
        by_doc = {doc: g for doc, g in eligible.groupby("document_id")}
        ok = lambda qid: f"m-{qid}" in validated_ok  # noqa: E731
        ordered = sorted((d for d in docs if d in by_doc), key=lambda d: (not by_doc[d].question_id.map(ok).any(),
                                                                         docs.index(d)))
        out = []
        for doc in ordered:
            g = by_doc[doc]
            first = sorted(hashed(g.question_id, "q"), key=lambda qid: not ok(qid))[0]
            out.append(g[g.question_id == first].iloc[0])
            taken.add(f"m-{first}")
            if len(out) == n:
                return out
        raise ValueError(f"only {len(out)}/{n} questions available")

    def docs_at(scale: int) -> list[str]:
        return hashed(q0[q0.min_scale == scale].document_id.unique(), f"s{scale}")

    split1_docs = hashed(set(q1.document_id) - set(manifest), "split1")
    q1 = q1.assign(answerable=q1.answer.str.strip() != NOT_FOUND)

    pools = {
        "s300": lambda n, a: pick(q0, answerable_docs + rest_docs, n, a),
        # all S300 docs, unused ones first: ~2/3 of RepliQA's "not found" questions fail corpus-level validation
        "s300_rest": lambda n, a: pick(q0, rest_docs + answerable_docs, n, a),
        "split1": lambda n, a: pick(q1, split1_docs, n, True),  # answerable in its own doc, doc absent here
        "s900_only": lambda n, a: pick(q0, docs_at(900), n, a),
        "s3000_only": lambda n, a: pick(q0, docs_at(3000), n, a),
    }

    items = []
    for cat, (n, pool, scales, answerable) in MAIN_PLAN.items():
        for r in pools[pool](n, answerable):
            in_corpus = r.document_id in manifest
            items.append({
                "qid": f"m-{r.question_id}",
                "track": "main",
                "category": cat,
                "answerable": answerable,
                "scales": scales,
                "question_en": r.question,
                "answer": r.answer if answerable else NOT_FOUND,
                "long_answer": r.long_answer if answerable else "",
                "gold_doc_ids": [r.document_id] if in_corpus else [],
                "gold_paths": [manifest[r.document_id]["path"]] if in_corpus else [],
                "source_doc_id": r.document_id,
            })
    return items


def build_side(rng: random.Random) -> list[dict]:
    manifest = load_jsonl(ROOT / "corpora/multihop/docs.jsonl")
    by_url = {d["url"]: d for d in manifest}
    by_title = {d["title"]: d for d in manifest}
    queries = json.loads((RAW / "multihoprag/MultiHopRAG.json").read_text())

    items = []
    for qtype, n in SIDE_PLAN.items():
        pool = [(i, q) for i, q in enumerate(queries) if q["question_type"] == qtype]
        for i, q in rng.sample(pool, n):
            gold = []
            for ev in q["evidence_list"]:
                d = by_url.get(ev.get("url")) or by_title.get(ev.get("title", "").strip())
                if d and d["path"] not in [g["path"] for g in gold]:
                    gold.append(d)
            answerable = qtype != "null_query"
            if answerable and len(gold) < len({ev.get("url") for ev in q["evidence_list"]}):
                raise ValueError(f"unmapped evidence for side query {i}")
            items.append({
                "qid": f"s-{i:04d}",
                "track": "side",
                "category": qtype,
                "answerable": answerable,
                "scales": [609],
                "question_en": q["query"],
                "answer": q["answer"],
                "long_answer": "",
                "gold_doc_ids": [d["doc_id"] for d in gold],
                "gold_paths": [d["path"] for d in gold],
                "evidence_facts": [ev.get("fact", "") for ev in q["evidence_list"]],
            })
    return items


def write(items: list[dict], path: Path) -> None:
    """Translations and validation results live in per-qid caches (questions/*_cache.jsonl) owned by
    translate_questions.py / validate_questions.py, and are re-applied here."""
    path.parent.mkdir(exist_ok=True)
    tcache = load_cache(TRANSLATION_CACHE)
    vcache = load_cache(VALIDATION_CACHE)
    with open(path, "w", encoding="utf-8") as f:
        for q in items:
            t = tcache.get(q["qid"])
            q["question"] = (t["question"] if t else None) if q["lang"] == "zh" else q["question_en"]
            if q["lang"] == "zh" and t and t.get("translation_issues"):
                q["translation_issues"] = t["translation_issues"]
            if q["qid"] in vcache:
                apply_validation(q, vcache[q["qid"]]["validation"])
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


TRANSLATION_CACHE = ROOT / "questions/translation_cache.jsonl"
VALIDATION_CACHE = ROOT / "questions/validation_cache.jsonl"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["qid"]: r for r in map(json.loads, path.read_text().splitlines())}  # last entry wins


def append_cache(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_validation(q: dict, v: dict) -> None:
    q["validation"] = v
    if v["status"] == "ok" and q["answerable"]:
        alt = [a["path"] for a in v["answering"] if a["consistent_with_reference"]]
        if alt:
            q["alt_gold_paths"] = alt
            q["gold_paths"] = sorted(set(q["gold_paths"]) | set(alt))


VAGUE_CACHE = ROOT / "questions/vague_cache.jsonl"


def vague_variants(main: list[dict]) -> list[dict]:
    """answerable_vague items: latest non-excluded rewrite (scripts/build_vague.py) of each English answerable
    question. Paired with the original via source_qid."""
    if not VAGUE_CACHE.exists():
        return []
    latest: dict[str, dict] = {}
    for r in map(json.loads, VAGUE_CACHE.read_text().splitlines()):
        latest[r["source_qid"]] = r  # attempts are appended in order
    out = []
    for q in main:
        r = latest.get(q["qid"])
        if q["category"] != "answerable" or q["lang"] != "en" or not r or r["qid"] in EXCLUDED:
            continue
        out.append({**{k: q[k] for k in ("track", "answerable", "scales", "answer", "long_answer", "gold_doc_ids",
                                         "gold_paths", "source_doc_id")},
                    "qid": r["qid"], "category": "answerable_vague", "lang": "en_vague",
                    "question_en": r["question"], "source_qid": q["qid"]})
    return out


def seed_caches_from(path: Path) -> None:
    """One-off migration: move translations/validations embedded in an existing question file into the caches."""
    if not path.exists():
        return
    tcache, vcache = load_cache(TRANSLATION_CACHE), load_cache(VALIDATION_CACHE)
    for q in map(json.loads, path.read_text().splitlines()):
        if q.get("lang") == "zh" and q.get("question") and q["qid"] not in tcache:
            append_cache(TRANSLATION_CACHE, {"qid": q["qid"], "question_en": q["question_en"], "question": q["question"],
                                             "translation_issues": q.get("translation_issues", [])})
        if "validation" in q and q["qid"] not in vcache:
            append_cache(VALIDATION_CACHE, {"qid": q["qid"], "validation": q["validation"]})


if __name__ == "__main__":
    for f in ("main.jsonl", "side.jsonl"):
        seed_caches_from(ROOT / "questions" / f)
    rng = random.Random(SEED)
    main, side = build_main(rng), build_side(rng)
    assign_lang(main)
    assign_lang(side)
    main += vague_variants(main)
    write(main, ROOT / "questions/main.jsonl")
    write(side, ROOT / "questions/side.jsonl")
    for name, items in (("main", main), ("side", side)):
        print(name, len(items), dict(Counter((q["category"], q["lang"]) for q in items)))
