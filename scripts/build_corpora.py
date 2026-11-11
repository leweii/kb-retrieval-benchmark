"""Build markdown corpora from the public datasets.

RepliQA split 0  -> corpora/repliqa/S{300,900,3000}/raw/<topic>/<title-slug>.md  (nested: S300 ⊂ S900 ⊂ S3000)
MultiHop-RAG     -> corpora/multihop/raw/<category>/<title-slug>.md

Each corpus dir gets a docs.jsonl manifest mapping dataset doc ids to file paths.
Deterministic: fixed seed, rerunnable (output dirs are rebuilt from scratch).
"""

from __future__ import annotations

import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "corpora"
SEED = 42
SCALES = (300, 900, 3000)

TERMINAL = tuple('.!?"”’:;,)')


def slugify(text: str, max_len: int = 70) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if len(text) > max_len:
        text = text[:max_len].rsplit("-", 1)[0]
    return text or "untitled"


def yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # JSON strings are valid YAML scalars


def frontmatter(fields: dict[str, str]) -> str:
    lines = ["---"] + [f"{k}: {yaml_str(v)}" for k, v in fields.items() if v] + ["---"]
    return "\n".join(lines)


def is_heading(block: str) -> bool:
    return "\n" not in block and len(block) <= 90 and not block.endswith(TERMINAL) and len(block.split()) >= 2


def first_sentence_title(block: str, max_len: int = 80) -> str:
    sent = re.split(r"(?<=[.!?])\s", block, maxsplit=1)[0]
    if len(sent) <= max_len:
        return sent.rstrip(".")
    return sent[:max_len].rsplit(" ", 1)[0]


SMALL = {"a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "as", "from",
         "into", "through", "vs", "&", "-", "–", "—"}


def split_fused(block: str, min_words: int = 4, max_words: int = 16) -> tuple[str, str] | None:
    """PDF extraction often fuses a Title-Case heading with the next sentence:
    'Unwrapping the Festive Flavors During the festive season, ...'
    Take the leading Title-Case run, drop the capitalized word that starts the sentence and any trailing
    small words. Returns (heading, rest) or None. Never drops text: heading + rest == block."""
    tokens = block.split(" ")
    run = 0
    for tok in tokens[: max_words + 2]:
        word = tok.strip(":")
        if word.lower() in SMALL or (word[:1].isupper() and not tok.endswith((",", ".", ";", "!", "?"))):
            run += 1
        else:
            break
    if run >= len(tokens) or run > max_words + 1:
        return None
    # Walk back over lowercase small words ('the', 'a') to the capitalized sentence starter and drop it,
    # then strip any trailing small words ('On', 'and') left at the end of the heading.
    end = run
    while end > 0 and tokens[end - 1].lower() in SMALL and not tokens[end - 1][:1].isupper():
        end -= 1
    end -= 1
    while end > 0 and tokens[end - 1].lower() in SMALL:
        end -= 1
    heading = tokens[:end]
    if len(heading) < min_words or not heading[-1][:1].isupper() or tokens[0][:1] in "\"“'‘(":
        return None
    return " ".join(heading).rstrip(":"), " ".join(tokens[end:])


GENERIC_TITLES = {"introduction", "company overview", "for immediate release", "title:", "# introduction #"}


def repliqa_to_md(text: str) -> tuple[str, str]:
    """Return (title, markdown body). Rejoins PDF line breaks; promotes standalone short lines and
    fused Title-Case prefixes to ## headings."""
    blocks = [" ".join(l.strip() for l in b.split("\n") if l.strip()) for b in re.split(r"\n\s*\n", text)]
    blocks = [b for b in blocks if b]

    body: list[str] = []
    for b in blocks:
        if is_heading(b):
            body.append(f"## {b}")
        elif (fused := split_fused(b)) is not None:
            body += [f"## {fused[0]}", fused[1]]
        else:
            body.append(b)

    title = None
    if body[0].startswith("## ") and body[0][3:].strip().lower() not in GENERIC_TITLES:
        title = body.pop(0)[3:]
    if title is None:  # no usable heading: first sentence of the first substantial prose paragraph
        prose = [b for b in body if not b.startswith("## ")]
        title = first_sentence_title(next((b for b in prose if len(b.split()) >= 5), prose[0]))
    return title, f"# {title}\n\n" + "\n\n".join(body) + "\n"


def unique_path(base: Path, used: set[Path]) -> Path:
    path, n = base, 2
    while path in used:
        path = base.with_name(f"{base.stem}-{n}{base.suffix}")
        n += 1
    used.add(path)
    return path


def write_corpus(dest: Path, docs: list[dict]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for d in docs:
        p = dest / "raw" / d["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d["md"], encoding="utf-8")
    with open(dest / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({k: v for k, v in d.items() if k != "md"}, ensure_ascii=False) + "\n")


def build_repliqa() -> None:
    df = pd.read_parquet(RAW / "repliqa" / "repliqa_0.parquet").drop_duplicates("document_id")
    ids = sorted(df.document_id)
    random.Random(SEED).shuffle(ids)
    rank = {doc_id: i for i, doc_id in enumerate(ids)}
    rows = df.set_index("document_id")

    used: set[Path] = set()
    docs = []
    for doc_id in ids[: max(SCALES)]:
        r = rows.loc[doc_id]
        title, body = repliqa_to_md(r.document_extracted)
        topic = r.document_topic
        rel = unique_path(Path(slugify(topic)) / f"{slugify(title)}.md", used)
        md = frontmatter({"title": title, "topic": topic}) + "\n\n" + body
        min_scale = next(s for s in SCALES if rank[doc_id] < s)
        docs.append({"doc_id": doc_id, "path": str(rel), "title": title, "topic": topic,
                     "min_scale": min_scale, "md": md})

    for s in SCALES:
        subset = [d for d in docs if d["min_scale"] <= s]
        write_corpus(OUT / "repliqa" / f"S{s}", subset)
        print(f"repliqa S{s}: {len(subset)} docs")


def build_multihop() -> None:
    corpus = json.loads((RAW / "multihoprag" / "corpus.json").read_text())
    used: set[Path] = set()
    docs = []
    for i, d in enumerate(corpus):
        title = d["title"].strip()
        body = re.sub(r"\n{3,}", "\n\n", d["body"].strip())
        fields = {"title": title, "source": d.get("source", ""), "author": d.get("author", ""),
                  "category": d.get("category", ""), "published_at": d.get("published_at", ""),
                  "url": d.get("url", "")}
        rel = unique_path(Path(slugify(d.get("category") or "misc")) / f"{slugify(title)}.md", used)
        md = frontmatter(fields) + f"\n\n# {title}\n\n{body}\n"
        docs.append({"doc_id": f"mh{i:04d}", "path": str(rel), "title": title, "url": d.get("url", ""),
                     "category": d.get("category", ""), "md": md})
    write_corpus(OUT / "multihop", docs)
    print(f"multihop: {len(docs)} docs")


if __name__ == "__main__":
    build_repliqa()
    build_multihop()
