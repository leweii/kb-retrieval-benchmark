"""Corpus loading and markdown chunking shared by all index-based retrievers.

Chunking: split on markdown headings, then greedily merge consecutive sections up to TARGET_TOKENS;
sections longer than MAX_TOKENS are split on paragraph boundaries. Every chunk is prefixed with the doc
title and its heading path so it stays interpretable out of context. Frontmatter is not indexed as text
(title is already in the prefix).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_TOKENS = 600
MAX_TOKENS = 900


@dataclass
class Chunk:
    chunk_id: str
    path: str  # doc path relative to the corpus raw/ dir
    title: str
    heading: str
    text: str  # chunk text including the title/heading prefix


def approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[一-鿿]", text))
    return int(len(re.findall(r"\S+", text)) * 1.3 + cjk)


def split_frontmatter(md: str) -> tuple[dict, str]:
    if md.startswith("---\n"):
        end = md.find("\n---", 4)
        if end != -1:
            meta = {}
            for line in md[4:end].splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    try:
                        meta[k] = json.loads(v)
                    except json.JSONDecodeError:
                        meta[k] = v
            return meta, md[end + 4 :].lstrip("\n")
    return {}, md


def sections(body: str) -> list[tuple[str, str]]:
    """[(heading, text)] split on #/##/### headings; the H1 title line is dropped."""
    out, heading, buf = [], "", []
    for line in body.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                out.append((heading, "\n".join(buf).strip()))
            buf = []
            heading = "" if len(m.group(1)) == 1 else m.group(2).strip()
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        out.append((heading, "\n".join(buf).strip()))
    return out


def split_long(text: str) -> list[str]:
    parts, cur = [], []
    for para in re.split(r"\n\s*\n", text):
        if cur and approx_tokens("\n\n".join(cur + [para])) > MAX_TOKENS:
            parts.append("\n\n".join(cur))
            cur = []
        cur.append(para)
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def chunk_doc(rel_path: str, md: str) -> list[Chunk]:
    meta, body = split_frontmatter(md)
    title = meta.get("title") or Path(rel_path).stem
    pieces: list[tuple[str, str]] = []
    for heading, text in sections(body):
        for part in split_long(text) if approx_tokens(text) > MAX_TOKENS else [text]:
            pieces.append((heading, part))

    merged: list[tuple[list[str], list[str]]] = []
    for heading, text in pieces:
        if merged and approx_tokens("\n\n".join(merged[-1][1] + [text])) <= TARGET_TOKENS:
            merged[-1][0].append(heading)
            merged[-1][1].append(f"## {heading}\n{text}" if heading else text)
        else:
            merged.append(([heading], [f"## {heading}\n{text}" if heading else text]))

    chunks = []
    for i, (headings, texts) in enumerate(merged):
        heading = " / ".join(h for h in headings if h)
        chunks.append(Chunk(f"{rel_path}#{i}", rel_path, title, heading, f"# {title}\n" + "\n\n".join(texts)))
    return chunks


def corpus_dir(name: str) -> Path:
    """name: 'repliqa/S300' | 'repliqa/S900' | 'repliqa/S3000' | 'multihop'; an '@wiki' suffix is accepted and
    stripped (the corpus root is the same, see doc_root)."""
    return ROOT / "corpora" / name.split("@")[0]


def doc_root(name: str) -> Path:
    """Directory whose .md files form the searchable documents: raw/ for raw corpora, wiki/ for '@wiki'
    (wiki pages only — the copied raw notes under wiki/raw/ are reachable by link, not indexed)."""
    return corpus_dir(name) / ("wiki" if name.endswith("@wiki") else "raw")


def load_chunks(name: str) -> list[Chunk]:
    root = doc_root(name)
    chunks: list[Chunk] = []
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root)
        if name.endswith("@wiki") and rel.parts[0] == "raw":
            continue
        chunks += chunk_doc(str(rel), p.read_text(encoding="utf-8"))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines()]
