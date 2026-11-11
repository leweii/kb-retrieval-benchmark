"""v2 step B: build the mixed knowledge base (S300 ⊂ S900 ⊂ S3000) and its question set.

Documents ("篇", one .md each):
  wiki/<slug>.md                  Wikipedia page = all MuSiQue dev paragraphs sharing a title
  bright/<domain>/<slug>.md       BRIGHT long document (scraped web page) from 7 StackExchange domains
  books/<book>/part-NNNN.md       ~500-word chunk of a NoLiMa (III-filter) book; needle sentences inserted
Questions (questions_v2/questions.jsonl):
  musique_ans / musique_unans     multi-hop; unans = one gold page (the last hop) removed from every corpus
  bright_ans / bright_unans       real forum questions; unans = all gold documents removed from every corpus
  nolima                          latent-association needle questions
  tier core -> asked at S300/S900/S3000 (gold docs in S300); tier ext -> asked at S900/S3000.
MuSiQue candidates are restricted to questions Claude Sonnet 5.5 could NOT answer closed-book.
Deterministic (hash orders); rebuilds output dirs from scratch.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_CORPORA = ROOT / "corpora" / "kb2"
OUT_Q = ROOT / "questions_v2"
SCALES = (300, 900, 3000)
SHARE = {"wiki": 0.55, "bright": 0.35, "books": 0.10}
PLAN = {  # category: (core, ext)
    "musique_ans": (60, 140), "musique_unans": (20, 30),
    "bright_ans": (40, 80), "bright_unans": (10, 20),
    "nolima": (5, 5),  # templates (not questions): 1-2 questions each
}
BOOK_WORDS = 500


def h(*parts) -> str:
    return hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()


def slugify(text: str, max_len: int = 70) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:max_len].rsplit("-", 1)[0] if len(s) > max_len else s) or "untitled"


def frontmatter(**fields) -> str:
    return "---\n" + "".join(f"{k}: {json.dumps(v, ensure_ascii=False)}\n" for k, v in fields.items() if v) + "---\n\n"


# ------------------------------------------------------------------ document pools

def wiki_pool() -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in open(RAW / "musique/musique_full_v1.0_dev.jsonl")]
    paras: dict[str, list[str]] = {}
    for r in rows:
        for p in r["paragraphs"]:
            lst = paras.setdefault(p["title"], [])
            if p["paragraph_text"] not in lst:
                lst.append(p["paragraph_text"])
    docs = {f"wiki:{t}": {"source": "wiki", "domain": "wikipedia", "title": t,
                           "body": "\n\n".join(ps)} for t, ps in paras.items()}
    return docs, rows


def bright_pool() -> tuple[dict, list[dict]]:
    docs, qs = {}, []
    for f in sorted((RAW / "bright/long_documents").glob("*.parquet")):
        dom = f.stem
        for r in pd.read_parquet(f).itertuples():
            docs[f"bright:{dom}/{r.id}"] = {"source": "bright", "domain": dom,
                                            "title": Path(r.id).stem.replace("_", " "), "body": r.content}
        for r in pd.read_parquet(RAW / "bright/examples" / f.name).itertuples():
            gold = [f"bright:{dom}/{g}" for g in r.gold_ids_long]
            if gold and all(g in docs for g in gold):
                qs.append({"id": f"{dom}/{r.id}", "domain": dom, "query": r.query, "gold": gold,
                           "gold_answer": r.gold_answer})
    return docs, qs


def book_pool() -> dict:
    docs = {}
    for f in sorted((RAW / "nolima/books/books/III-filter").glob("*.txt")):
        words = f.read_text(errors="ignore").split()
        name = f.stem.replace("_", " ").title()
        for i in range(0, len(words), BOOK_WORDS):
            n = i // BOOK_WORDS + 1
            docs[f"books:{f.stem}#{n}"] = {"source": "books", "domain": "fiction", "title": f"{name} — part {n}",
                                          "body": " ".join(words[i : i + BOOK_WORDS])}
    return docs


# ------------------------------------------------------------------ question selection

def select_musique(rows: list[dict], wiki: dict) -> tuple[list[dict], set[str]]:
    cb = {json.loads(l)["id"]: json.loads(l) for l in (OUT_Q / "closed_book_sonnet.jsonl").read_text().splitlines()}
    usable = sorted((r for r in rows if r["answerable"] and r["id"] in cb and not cb[r["id"]]["correct"]),
                    key=lambda r: h("musique", r["id"]))

    def gold_titles(r):
        return sorted({p["title"] for p in r["paragraphs"] if p["is_supporting"]})

    n_ans = sum(PLAN["musique_ans"])
    ans = usable[:n_ans]
    ans_gold = {t for r in ans for t in gold_titles(r)}
    unans, removed = [], set()
    for r in usable[n_ans:]:
        last = r["paragraphs"][r["question_decomposition"][-1]["paragraph_support_idx"]]["title"]
        if last in ans_gold or last in removed:
            continue
        unans.append((r, last))
        removed.add(last)
        if len(unans) == sum(PLAN["musique_unans"]):
            break
    if len(ans) < n_ans or len(unans) < sum(PLAN["musique_unans"]):
        raise SystemExit(f"not enough closed-book-unknown MuSiQue questions: ans {len(ans)}, unans {len(unans)} "
                         "-> run kb2_closed_book.py with more candidates")

    out = []
    for i, r in enumerate(ans):
        out.append({"src_id": r["id"], "category": "musique_ans", "tier": "core" if i < PLAN["musique_ans"][0] else "ext",
                    "question_en": r["question"], "answer": r["answer"], "answer_aliases": r.get("answer_aliases", []),
                    "gold_keys": [f"wiki:{t}" for t in gold_titles(r)], "hops": len(r["question_decomposition"]),
                    "hard_negatives": [f"wiki:{p['title']}" for p in r["paragraphs"] if not p["is_supporting"]]})
    for i, (r, last) in enumerate(unans):
        out.append({"src_id": r["id"], "category": "musique_unans",
                    "tier": "core" if i < PLAN["musique_unans"][0] else "ext",
                    "question_en": r["question"], "answer": None, "original_answer": r["answer"],
                    "gold_keys": [f"wiki:{t}" for t in gold_titles(r) if t != last],  # partial chain stays findable
                    "removed_keys": [f"wiki:{last}"], "hops": len(r["question_decomposition"]),
                    "hard_negatives": [f"wiki:{p['title']}" for p in r["paragraphs"] if not p["is_supporting"]]})
    return out, {f"wiki:{t}" for t in removed}


def select_bright(qs: list[dict]) -> tuple[list[dict], set[str]]:
    qs = sorted(qs, key=lambda q: h("bright", q["id"]))
    n_ans, n_un = sum(PLAN["bright_ans"]), sum(PLAN["bright_unans"])
    ans = qs[:n_ans]
    ans_gold = {g for q in ans for g in q["gold"]}
    unans, removed = [], set()
    for q in qs[n_ans:]:
        if set(q["gold"]) & ans_gold:
            continue
        unans.append(q)
        removed |= set(q["gold"])
        if len(unans) == n_un:
            break
    out = []
    for i, q in enumerate(ans):
        out.append({"src_id": q["id"], "category": "bright_ans", "tier": "core" if i < PLAN["bright_ans"][0] else "ext",
                    "question_en": q["query"], "answer": q["gold_answer"], "gold_keys": q["gold"], "domain": q["domain"]})
    for i, q in enumerate(unans):
        out.append({"src_id": q["id"], "category": "bright_unans", "tier": "core" if i < PLAN["bright_unans"][0] else "ext",
                    "question_en": q["query"], "answer": None, "original_answer": q["gold_answer"], "gold_keys": [],
                    "removed_keys": q["gold"], "domain": q["domain"]})
    return out, removed


def select_nolima(books: dict) -> list[dict]:
    """One test per needle template. NoLiMa puts a single needle in each haystack; in a shared knowledge base,
    several instances of one template conflict ("Which character cannot drink milk?" would match every
    "I'm lactose intolerant" needle; "has been to Germany?" matches both the Frankfurt and the Dresden test).
    The chosen test's needle goes into one book chunk; its one-hop and two-hop questions share that chunk.
    The first PLAN['nolima'][0] templates (hash order) are core, the rest ext."""
    needles = sorted(json.load(open(RAW / "nolima/needle_set.json")), key=lambda nd: h("nolima-template", nd["id"]))
    chunk_keys = sorted(books, key=lambda k: h("chunk", k))
    out = []
    for i, nd in enumerate(needles):
        tid = sorted(nd["tests"], key=lambda t: h("nolima-test", nd["id"], t))[0]
        args = nd["tests"][tid]["input_args"]
        rng = random.Random(h("char", nd["id"], tid))
        char = rng.choice(nd["character_set"])
        needle = nd["needle"].replace("{CHAR}", char)
        for j, a in enumerate(args, 1):
            needle = needle.replace("{" + str(j) + "}", a)
        key = chunk_keys[i]
        body = books[key]["body"]
        # insert at a real sentence boundary: ". " + capital letter, not after an abbreviation like "Mr."
        cuts = [m.end() for m in re.finditer(r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)[.!?][\"”’]? (?=[A-Z])", body)]
        cut = cuts[rng.randrange(len(cuts))] if cuts else len(body)
        books[key]["body"] = body[:cut] + needle.strip() + " " + body[cut:]
        for hop, qt in nd["questions"].items():
            q = qt
            for j, a in enumerate(args, 1):
                q = q.replace("{" + str(j) + "}", a)
            out.append({"src_id": f"{nd['id']}/{tid}/{hop}", "category": "nolima",
                        "tier": "core" if i < PLAN["nolima"][0] else "ext", "question_en": q, "answer": char,
                        "answer_aliases": [], "gold_keys": [key], "needle": needle,
                        "reasoning_type": nd["reasoning_type"], "hop": hop})
    return out


# ------------------------------------------------------------------ corpus assembly

def assemble(questions: list[dict], pools: dict[str, dict], removed: set[str]) -> dict[str, int]:
    """Return doc_key -> min_scale."""
    min_scale: dict[str, int] = {}
    core_gold = [g for q in questions if q["tier"] == "core" for g in q["gold_keys"]]
    ext_gold = [g for q in questions if q["tier"] == "ext" for g in q["gold_keys"]]
    needles_ext = {g for q in questions if q["category"] == "nolima" and q["tier"] == "ext" for g in q["gold_keys"]}
    for g in core_gold:
        min_scale.setdefault(g, 300)
    for g in ext_gold:
        min_scale.setdefault(g, 900)
    assert not (set(min_scale) & removed), "a removed doc is gold for another question"

    def src(k):
        return k.split(":", 1)[0]

    hard_core = [n for q in questions if q["tier"] == "core" for n in q.get("hard_negatives", [])]
    hard_all = hard_core + [n for q in questions if q["tier"] == "ext" for n in q.get("hard_negatives", [])]
    filler_order = {
        "wiki": list(dict.fromkeys(hard_all + sorted(pools["wiki"], key=lambda k: h("fill", k)))),
        "bright": sorted(pools["bright"], key=lambda k: h("fill", k)),
        "books": sorted(pools["books"], key=lambda k: h("fill", k)),
    }
    hard_core_set = set(hard_core)
    for scale in SCALES:
        included = [k for k, s in min_scale.items() if s <= scale]
        have = Counter(src(k) for k in included)
        need = scale - len(included)
        if need < 0:
            raise SystemExit(f"S{scale}: gold docs alone ({len(included)}) exceed scale")
        target = {s: round(scale * SHARE[s]) for s in SHARE}
        deficit = {s: max(0, target[s] - have[s]) for s in SHARE}
        total_def = sum(deficit.values()) or 1
        quota = {s: round(need * deficit[s] / total_def) for s in SHARE}
        quota["wiki"] += need - sum(quota.values())  # rounding remainder
        for s, n in quota.items():
            order = filler_order[s]
            if scale == 300 and s == "wiki":  # S300 fillers: hard negatives of core questions first
                order = sorted(order, key=lambda k: k not in hard_core_set)
            added = 0
            for k in order:
                if added >= n:
                    break
                if k in min_scale or k in removed or k in needles_ext or k not in pools[s]:
                    continue
                min_scale[k] = scale
                added += 1
            if added < n:
                raise SystemExit(f"S{scale}: ran out of {s} fillers")
    return min_scale


def write_corpora(min_scale: dict[str, int], pools: dict[str, dict]) -> dict[str, str]:
    paths, used = {}, set()
    for k in sorted(min_scale, key=lambda k: (min_scale[k], k)):
        d = pools[k.split(":", 1)[0]][k]
        if d["source"] == "wiki":
            base = f"wiki/{slugify(d['title'])}"
        elif d["source"] == "bright":
            base = f"bright/{d['domain']}/{slugify(d['title'])}"
        else:
            book, n = k[6:].split("#")
            base = f"books/{slugify(book)}/part-{int(n):04d}"
        p, i = base + ".md", 2
        while p in used:
            p, i = f"{base}-{i}.md", i + 1
        used.add(p)
        paths[k] = p
    for scale in SCALES:
        dest = OUT_CORPORA / f"S{scale}"
        if dest.exists():
            shutil.rmtree(dest)
        rows = []
        for k, s in min_scale.items():
            if s > scale:
                continue
            d = pools[k.split(":", 1)[0]][k]
            f = dest / "raw" / paths[k]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(frontmatter(title=d["title"], source=d["source"], domain=d["domain"]) + f"# {d['title']}\n\n"
                         + d["body"].strip() + "\n", encoding="utf-8")
            rows.append({"doc_key": k, "path": paths[k], "title": d["title"], "source": d["source"],
                         "domain": d["domain"], "min_scale": s})
        (dest / "docs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"S{scale}: {len(rows)} docs {dict(Counter(r['source'] for r in rows))}")
    return paths


def main() -> None:
    wiki, mrows = wiki_pool()
    bright, bqs = bright_pool()
    books = book_pool()
    pools = {"wiki": wiki, "bright": bright, "books": books}

    mus, removed_w = select_musique(mrows, wiki)
    bri, removed_b = select_bright(bqs)
    nol = select_nolima(books)
    questions = mus + bri + nol
    removed = removed_w | removed_b
    min_scale = assemble(questions, pools, removed)
    paths = write_corpora(min_scale, pools)

    by_cat = {}
    for q in questions:
        by_cat.setdefault((q["category"], q["tier"]), []).append(q)
    out = []
    for (cat, tier), qs in sorted(by_cat.items()):
        qs.sort(key=lambda q: h("lang", q["src_id"]))
        for i, q in enumerate(qs):
            out.append({
                "qid": f"{cat}-{h(q['src_id'])[:10]}", "track": cat.split("_")[0], **q,
                "scales": [300, 900, 3000] if tier == "core" else [900, 3000],
                "answerable": q["answer"] is not None,
                "lang": "zh" if i < len(qs) // 2 else "en",
                "gold_paths": [paths[g] for g in q["gold_keys"]],
                "removed_doc_keys": q.get("removed_keys", []),
            })
    for q in out:
        q.pop("hard_negatives", None)
        q.pop("removed_keys", None)
    OUT_Q.mkdir(exist_ok=True)
    (OUT_Q / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out))
    print("questions:", len(out), dict(Counter((q["category"], q["tier"]) for q in out)))
    print("removed docs (never in any corpus):", len(removed))


if __name__ == "__main__":
    main()
