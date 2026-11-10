"""Digest raw corpora into an LLM-wiki (Karpathy-style): source pages + entity pages + topic pages + index.md,
cross-linked with [[wikilinks]].

  map    (LLM, cached per doc)  Claude Sonnet 5.5 reads one raw note and writes its wiki fields:
                                summary, key facts, entities, tags.          -> data/digest/<dataset>/<path>.json
  reduce (code, per corpus)     builds corpora/<corpus>/wiki/ from the cached map outputs of that corpus's docs.

Wiki layout (paths relative to corpora/<corpus>/wiki/):
  index.md                      topics with doc counts -> topic pages
  topics/<topic>.md             every source page in the topic with a one-line summary
  sources/<topic>/<slug>.md     summary + key facts + [[entity]] links + link to the raw note
  entities/<slug>.md            only for entities mentioned in >= 2 sources: facts per source, backlinks

Usage: digest_wiki.py map <corpus> [--limit N]      e.g. map repliqa/S3000   (S300/S900 reuse its cache)
       digest_wiki.py reduce <corpus> [...]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrievers.corpus import corpus_dir, split_frontmatter  # noqa: E402

MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.00, 10.00
EFFORT = "medium"
COMPACT_ABOVE = 15  # entity pages with more mentions list sources + roles only

SYSTEM = """You maintain a personal knowledge wiki in markdown. For each source note you write the fields of its \
wiki page so the note can be found and understood later without rereading it.

- summary: 2-4 sentences on what the note is about and its main points.
- one_line: one sentence (max ~25 words) for the topic catalog.
- key_facts: 8-15 self-contained facts. Keep specific names, numbers, dates, quantities, places and short \
quotes exactly as in the note. Each fact must make sense on its own (repeat the subject's name instead of "he").
- entities: the people, organizations, places, events, products and programs the note names, each with a \
short role description as it appears in this note. Use the full name as written in the note.
- tags: 3-6 lowercase topical tags.
- category: the single best-fitting broad category from the given list.
The note may be a clipped web page with navigation menus, cookie banners, scripts or JSON — ignore that \
boilerplate and digest only the actual content. Only use information from the note."""

CATEGORIES = ["people", "places & geography", "sports", "music", "film & television", "literature & arts",
              "history & politics", "government & law", "business & economy", "science & technology",
              "health & psychology", "religion & culture", "other"]
MAX_INPUT_CHARS = 32000  # long scraped pages: the first ~8k tokens carry the content

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "one_line": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["person", "organization", "place", "event", "product",
                                                         "program", "other"]},
                    "role": {"type": "string"},
                },
                "required": ["name", "type", "role"],
                "additionalProperties": False,
            },
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string", "enum": CATEGORIES},
    },
    "required": ["summary", "one_line", "key_facts", "entities", "tags", "category"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=6)


MODELS = {  # --model name -> (model id, $/M in, $/M out, cache dir suffix)
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00, ""),
    "haiku": ("claude-haiku-5-5", 0.10, 0.50, "-haiku"),
}
CACHE_SUFFIX = ""


def set_model(name: str) -> None:
    global MODEL, PRICE_IN, PRICE_OUT, CACHE_SUFFIX
    MODEL, PRICE_IN, PRICE_OUT, CACHE_SUFFIX = MODELS[name]


def dataset_of(corpus: str) -> str:
    return corpus.split("/")[0]


def cache_path(corpus: str, rel: str) -> Path:
    return ROOT / "data" / "digest" / (dataset_of(corpus) + CACHE_SUFFIX) / (rel + ".json")


def digest_one(corpus: str, rel: str) -> float:
    out = cache_path(corpus, rel)
    if out.exists():
        return 0.0
    md = (corpus_dir(corpus) / "raw" / rel).read_text(encoding="utf-8")[:MAX_INPUT_CHARS]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": md}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"{rel}: stop_reason={resp.stop_reason}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    data["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return (resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT) / 1e6


def run_map(corpus: str, limit: int | None) -> None:
    docs = [json.loads(l) for l in (corpus_dir(corpus) / "docs.jsonl").read_text().splitlines()]
    if "min_scale" in docs[0]:  # process in nesting order so a partial run still covers S300 first
        docs.sort(key=lambda d: d["min_scale"])
    todo = [d["path"] for d in docs if not cache_path(corpus, d["path"]).exists()][:limit]
    print(f"{corpus}: {len(todo)} docs to digest ({len(docs) - len(todo)} cached / skipped)", flush=True)
    done, cost = 0, 0.0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for c in pool.map(lambda rel: digest_one(corpus, rel), todo):
            done += 1
            cost += c
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}  ${cost:.2f}", flush=True)
    print(f"{corpus}: map done, ${cost:.2f}")


# ---------------------------------------------------------------- reduce

def slug(text: str, max_len: int = 70) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:max_len].rsplit("-", 1)[0] if len(s) > max_len else s) or "x"


def topic_of(meta: dict, data: dict) -> str:
    """v1 corpora carry topic/category in frontmatter. v2 (kb2): encyclopedia pages are grouped by the
    LLM-assigned category (1650 pages under one 'wikipedia' topic would be unnavigable), web clippings by
    StackExchange domain, fiction chunks by book."""
    if meta.get("topic") or meta.get("category"):
        return meta.get("topic") or meta.get("category")
    src = meta.get("source")
    if src == "wiki":
        return f"encyclopedia: {data.get('category', 'other')}"
    if src == "bright":
        return f"web clippings: {meta.get('domain', 'misc').replace('_', ' ')}"
    if src == "books":
        return f"fiction: {meta.get('title', 'book').split(' — ')[0]}"
    return "misc"


def norm_entity(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^(the|dr\.?|mr\.?|mrs\.?|ms\.?)\s+", "", name.strip(), flags=re.I)).casefold()


def run_reduce(corpus: str) -> None:
    cdir = corpus_dir(corpus)
    docs = [json.loads(l) for l in (cdir / "docs.jsonl").read_text().splitlines()]
    wiki = cdir / "wiki"
    if wiki.exists():
        shutil.rmtree(wiki)

    pages, used_pages = [], set()
    for d in docs:
        cp = cache_path(corpus, d["path"])
        if not cp.exists():
            raise FileNotFoundError(f"missing digest for {d['path']} — run map first")
        meta, _ = split_frontmatter((cdir / "raw" / d["path"]).read_text(encoding="utf-8"))
        data = json.loads(cp.read_text())
        topic = topic_of(meta, data)
        page, i = f"sources/{slug(topic)}/{Path(d['path']).stem}", 2
        while page in used_pages:
            page, i = f"sources/{slug(topic)}/{Path(d['path']).stem}-{i}", i + 1
        used_pages.add(page)
        pages.append({"doc": d, "meta": meta, "topic": topic, "data": data, "page": page})

    # entity table: normalized name -> mentions
    mentions: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for p in pages:
        seen = set()
        for e in p["data"]["entities"]:
            k = norm_entity(e["name"])
            if k and k not in seen:
                seen.add(k)
                mentions[k].append((p, e))
    entity_page = {k: f"entities/{slug(v[0][1]['name'])}" for k, v in mentions.items() if len(v) >= 2}
    used = defaultdict(int)  # disambiguate slug collisions
    for k in sorted(entity_page):
        base = entity_page[k]
        used[base] += 1
        if used[base] > 1:
            entity_page[k] = f"{base}-{used[base]}"

    def write(rel: str, text: str) -> None:
        f = wiki / (rel + ".md")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")

    # source pages
    for p in pages:
        d, doc = p["data"], p["doc"]  # digest fields, manifest row
        title = doc["title"]
        ents = []
        for e in d["entities"]:
            k = norm_entity(e["name"])
            link = f"[[{entity_page[k]}|{e['name']}]]" if k in entity_page else e["name"]
            ents.append(f"- {link} ({e['type']}) — {e['role']}")
        meta_lines = ["---", f"title: {json.dumps(title, ensure_ascii=False)}", f"type: source",
                      f"topic: {json.dumps(p['topic'])}", f"tags: {json.dumps(d['tags'], ensure_ascii=False)}",
                      f"raw: {json.dumps('raw/' + doc['path'])}", "---"]
        body = [f"# {title}", "", d["summary"], "", "## Key facts", *[f"- {x}" for x in d["key_facts"]], "",
                "## Entities", *ents, "", f"Topic: [[topics/{slug(p['topic'])}|{p['topic']}]] · "
                f"Raw note: `raw/{doc['path']}`", ""]
        write(p["page"], "\n".join(meta_lines + [""] + body))

    # entity pages
    for k, page in entity_page.items():
        ms = mentions[k]
        name = ms[0][1]["name"]
        types = sorted({e["type"] for _, e in ms})
        lines = ["---", f"title: {json.dumps(name, ensure_ascii=False)}", "type: entity",
                 f"entity_type: {json.dumps('/'.join(types))}", f"mentions: {len(ms)}", "---", "",
                 f"# {name}", "", f"Mentioned in {len(ms)} sources. Different sources may refer to different "
                 f"{'/'.join(types)}s with the same name.", ""]
        tokens = [t for t in re.split(r"\W+", name.lower()) if len(t) > 2]
        if len(ms) > COMPACT_ABOVE:  # very common names: one line per source keeps the page readable
            lines += ["## Sources"] + [f"- [[{p['page']}|{p['doc']['title']}]] — {e['role']}" for p, e in ms] + [""]
        else:
            for p, e in ms:
                lines += [f"## [[{p['page']}|{p['doc']['title']}]]", f"Role: {e['role']}"]
                facts = [f for f in p["data"]["key_facts"] if any(t in f.lower() for t in tokens)]
                lines += [f"- {f}" for f in facts[:6]] + [""]
        write(page, "\n".join(lines))

    # topic pages + index
    by_topic = defaultdict(list)
    for p in pages:
        by_topic[p["topic"]].append(p)
    for topic, ps in by_topic.items():
        lines = ["---", f"title: {json.dumps(topic)}", "type: topic", f"sources: {len(ps)}", "---", "", f"# {topic}",
                 "", f"{len(ps)} sources. [[index|Back to index]]", ""]
        lines += [f"- [[{p['page']}|{p['doc']['title']}]] — {p['data']['one_line']}"
                  for p in sorted(ps, key=lambda p: p["doc"]["title"])]
        write(f"topics/{slug(topic)}", "\n".join(lines) + "\n")
    lines = ["---", "title: Index", "type: index", "---", "", "# Knowledge base index", "",
             f"{len(pages)} sources in {len(by_topic)} topics, {len(entity_page)} entity pages "
             "(entities that appear in 2+ sources).", "", "## Topics"]
    lines += [f"- [[topics/{slug(t)}|{t}]] ({len(ps)} sources)" for t, ps in sorted(by_topic.items())]
    lines += ["", "## How to navigate",
              "- Topic pages list every source with a one-line summary.",
              "- Source pages hold a summary, key facts and entities; each links to its raw note under `raw/`.",
              "- Entity pages collect facts about a name across sources."]
    write("index", "\n".join(lines) + "\n")

    # copy raw notes alongside so wiki links to raw/ resolve inside the wiki dir
    shutil.copytree(cdir / "raw", wiki / "raw")
    n_files = sum(1 for _ in wiki.rglob("*.md"))
    print(f"{corpus}: wiki with {len(pages)} source pages, {len(entity_page)} entity pages, "
          f"{len(by_topic)} topics ({n_files} md files incl. raw)")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if "--model" in args:  # sonnet (default) | haiku — selects model and cache dir for map and reduce
        i = args.index("--model")
        set_model(args[i + 1])
        args = args[:i] + args[i + 2:]
    if cmd == "map":
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
        run_map(args[0], limit)
    elif cmd == "reduce":
        for c in args:
            run_reduce(c)
    else:
        raise SystemExit(__doc__)
