"""Knowledge-base tools per retrieval configuration, plus evidence tracking.

  R1 grep          raw/   tools: grep, list_dir, read_file
  R2 bm25          raw/   tools: search (BM25), read_file
  R3 hybrid        raw/   tools: search (BM25 + dense, RRF), read_file
  R4 enhanced      raw/   tools: search (QMD-style: expansion + RRF + rerank), read_file
  R5 wiki_nav      wiki/  tools: grep, list_dir, read_file      (start at index.md; raw notes under wiki/raw/)
  R6 wiki_hybrid   wiki/  tools: search (hybrid over wiki pages), list_dir, read_file

Every tool call records which documents it surfaced (retrieved) and opened (read). Wiki source pages are mapped
back to their raw note so recall is comparable across raw and wiki configurations.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from harness.llm import Tool
from retrievers.corpus import ROOT, split_frontmatter

READ_MAX_LINES, READ_MAX_CHARS = 200, 12000
GREP_MAX_LINES, GREP_LINE_CHARS = 60, 300
SNIPPET_CHARS = 1500

CONFIGS = {
    "R1_grep": {"root": "raw", "tools": ["grep", "list_dir", "read_file"]},
    "R2_bm25": {"root": "raw", "tools": ["search", "read_file"], "retriever": "bm25"},
    "R3_hybrid": {"root": "raw", "tools": ["search", "read_file"], "retriever": "hybrid"},
    "R4_enhanced": {"root": "raw", "tools": ["search", "read_file"], "retriever": "enhanced"},
    "R5_wiki_nav": {"root": "wiki", "tools": ["grep", "list_dir", "read_file"]},
    "R6_wiki_hybrid": {"root": "wiki", "tools": ["search", "list_dir", "read_file"], "retriever": "hybrid@wiki"},
}

TOOL_SPECS = {
    "search": Tool("search", "Search the knowledge base. Returns the most relevant passages with their file paths. "
                   "Use several different queries if the first results don't contain the answer.",
                   {"type": "object", "properties": {"query": {"type": "string"},
                                                     "k": {"type": "integer", "description": "results (default 8, max 20)"}},
                    "required": ["query"]}),
    "grep": Tool("grep", "Case-insensitive regex search over all files in the knowledge base. Returns matching "
                 "lines as path:line: text (long lines truncated).",
                 {"type": "object", "properties": {"pattern": {"type": "string"},
                                                   "path": {"type": "string", "description": "subdirectory (optional)"}},
                  "required": ["pattern"]}),
    "list_dir": Tool("list_dir", "List files and subdirectories of a directory in the knowledge base.",
                     {"type": "object", "properties": {"path": {"type": "string", "description": "default: root"}}}),
    "read_file": Tool("read_file", f"Read a file from the knowledge base. Returns up to {READ_MAX_LINES} lines "
                      "starting at `offset` (0-based line number); call again with a larger offset to continue.",
                      {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
                       "required": ["path"]}),
}


import anthropic  # noqa: E402
import httpx  # noqa: E402
import openai  # noqa: E402

INFRA_ERRORS = (openai.APIError, anthropic.APIError, httpx.HTTPError, ConnectionError, TimeoutError)

_cache: dict = {}
_lock = threading.Lock()


def _cached(key, factory):
    with _lock:  # indexes are loaded once per process and shared by all concurrent tasks (read-only)
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def shared_retriever(corpus: str, kind: str):
    from retrievers.bm25 import BM25Index
    from retrievers.dense import DenseIndex
    from retrievers.hybrid import EnhancedHybrid, Hybrid

    target = corpus + "@wiki" if kind.endswith("@wiki") else corpus
    bm = _cached(("bm25", target), lambda: BM25Index(target))
    if kind == "bm25":
        return bm
    de = _cached(("dense", target), lambda: DenseIndex(target))
    if kind == "enhanced":
        return _cached(("enhanced", target), lambda: EnhancedHybrid(target, bm25=bm, dense=de))
    return _cached(("hybrid", target), lambda: Hybrid(target, bm25=bm, dense=de))


def file_lines(root: Path) -> list[tuple[str, list[str]]]:
    """All .md files under root as (relative path, lines), loaded once and shared."""
    def build():
        return [(str(p.relative_to(root)), p.read_text(encoding="utf-8", errors="ignore").splitlines())
                for p in sorted(root.rglob("*.md"))]
    return _cached(("lines", str(root)), build)


def wiki_map(corpus: str) -> dict[str, str]:
    """wiki source page path -> raw doc path (from the page's `raw:` frontmatter)."""
    def build():
        root = ROOT / "corpora" / corpus / "wiki"
        m = {}
        for p in (root / "sources").rglob("*.md"):
            meta, _ = split_frontmatter(p.read_text(encoding="utf-8"))
            if meta.get("raw"):
                m[str(p.relative_to(root))] = meta["raw"].removeprefix("raw/")
        return m
    return _cached(("wikimap", corpus), build)


@dataclass
class Evidence:
    retrieved: list[str] = field(default_factory=list)  # raw doc paths surfaced by search / grep
    read: list[str] = field(default_factory=list)  # raw doc paths opened (wiki source page or raw note)
    read_raw: list[str] = field(default_factory=list)  # raw notes opened directly (wiki configs: drill-down)
    tool_calls: int = 0
    retrieval_cost: float = 0.0
    infra_errors: list[str] = field(default_factory=list)

    def _add(self, lst: list[str], p: str | None) -> None:
        if p and p not in lst:
            lst.append(p)


class KB:
    def __init__(self, corpus: str, config: str):
        self.corpus, self.config = corpus, config
        self.cfg = CONFIGS[config]
        self.root = ROOT / "corpora" / corpus / self.cfg["root"]
        self.ev = Evidence()
        kind = self.cfg.get("retriever")
        self.retriever = (kind.split("@")[0], shared_retriever(corpus, kind)) if kind else None
        self.wiki_to_raw = wiki_map(corpus) if self.cfg["root"] == "wiki" else {}

    def raw_path(self, rel: str) -> str | None:
        """Map a path relative to the KB root to the raw doc path it represents (None for index/topic/entity)."""
        if self.cfg["root"] == "raw":
            return rel
        if rel.startswith("raw/"):
            return rel[4:]
        return self.wiki_to_raw.get(rel)

    def tools(self) -> list[Tool]:
        return [TOOL_SPECS[t] for t in self.cfg["tools"]]

    # ------------------------------------------------------------------ retrieval for A0 (single shot)

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        kind, r = self.retriever
        if kind == "bm25":
            hits = r.search(query, k)
        else:
            res = r.search(query, k)
            hits = res.hits
            self.ev.retrieval_cost += res.cost
        out = []
        for c, s in hits[:k]:
            self.ev._add(self.ev.retrieved, self.raw_path(c.path))
            out.append({"path": c.path, "heading": c.heading, "text": c.text})
        return out

    # ------------------------------------------------------------------ tool execution (A1)

    def _safe(self, rel: str | None) -> Path:
        p = (self.root / (rel or "").lstrip("/")).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError("path outside the knowledge base")
        return p

    def call(self, name: str, args: dict) -> str:
        self.ev.tool_calls += 1
        try:
            return getattr(self, f"_t_{name}")(**{k: v for k, v in args.items() if not k.startswith("_")})
        except TypeError as e:
            return f"[error] bad arguments for {name}: {e}"
        except INFRA_ERRORS as e:  # API/network failure of a retrieval backend: the run is invalid, not the model
            self.ev.infra_errors.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
            return f"[error] search backend unavailable: {type(e).__name__}"
        except Exception as e:  # model-caused tool errors (bad path, invalid regex) go back to the model
            return f"[error] {type(e).__name__}: {e}"

    def _t_search(self, query: str, k: int = 8) -> str:
        k = max(1, min(int(k or 8), 20))
        hits = self.retrieve(query, k)
        if not hits:
            return "No results."
        return "\n\n".join(f"[{i + 1}] {h['path']}" + (f" — {h['heading']}" if h["heading"] else "") + "\n"
                           + h["text"][:SNIPPET_CHARS] for i, h in enumerate(hits))

    def _t_grep(self, pattern: str, path: str | None = None) -> str:
        """Pure-Python grep (no ripgrep on this machine): case-insensitive regex, <=5 matches per file, long lines
        cut to GREP_LINE_CHARS around the match. File texts are cached in memory per knowledge base."""
        if len(pattern) > 300:
            return "[error] pattern too long"
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"[error] invalid regex: {e}"
        base = self._safe(path)
        prefix = "" if base == self.root.resolve() else str(base.relative_to(self.root.resolve())) + "/"
        out, total = [], 0
        for rel, lines in file_lines(self.root):
            if prefix and not rel.startswith(prefix):
                continue
            hits = 0
            for i, line in enumerate(lines):
                m = rx.search(line)
                if not m:
                    continue
                total += 1
                hits += 1
                if len(out) < GREP_MAX_LINES:
                    if len(line) > GREP_LINE_CHARS:
                        s = max(0, m.start() - GREP_LINE_CHARS // 2)
                        line = ("…" if s else "") + line[s : s + GREP_LINE_CHARS] + "…"
                    out.append(f"{rel}:{i + 1}: {line}")
                    self.ev._add(self.ev.retrieved, self.raw_path(rel))
                if hits >= 5:
                    break
        if not out:
            return "No matches."
        more = f"\n... {total - len(out)} more matching lines (refine the pattern)" if total > len(out) else ""
        return "\n".join(out) + more

    def _t_list_dir(self, path: str | None = None) -> str:
        d = self._safe(path)
        if not d.is_dir():
            return f"[error] not a directory: {path}"
        entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        head = lines[:300]
        return "\n".join(head) + (f"\n... {len(lines) - 300} more entries" if len(lines) > 300 else "")

    def _t_read_file(self, path: str, offset: int = 0) -> str:
        f = self._safe(path)
        if not f.is_file():
            return f"[error] no such file: {path}"
        rel = str(f.relative_to(self.root.resolve()))
        raw = self.raw_path(rel)
        self.ev._add(self.ev.read, raw)
        if self.cfg["root"] == "wiki" and rel.startswith("raw/"):
            self.ev._add(self.ev.read_raw, raw)
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        offset = max(0, int(offset or 0))
        chunk, size = [], 0
        for i in range(offset, min(len(lines), offset + READ_MAX_LINES)):
            line = lines[i][:READ_MAX_CHARS]
            if size + len(line) > READ_MAX_CHARS and chunk:
                break
            chunk.append(f"{i + 1}\t{line}")
            size += len(line)
        end = offset + len(chunk)
        tail = f"\n[lines {offset + 1}-{end} of {len(lines)}; continue with offset={end}]" if end < len(lines) else ""
        return "\n".join(chunk) + tail
