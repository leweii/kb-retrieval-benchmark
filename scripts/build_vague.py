"""Vague English variants of the English answerable questions (category answerable_vague, lang en_vague).

Real users often don't remember exact names. Claude Sonnet 5.5 rewrites each question replacing exact names of
people / organizations / products and coined terms with descriptive references taken from the gold document,
while keeping it specific enough to single out that document. Each attempt gets its own qid (v-<orig>-a<n>) so
validate_questions.py can check it; an attempt excluded as ambiguous is rewritten with that feedback (max 3).

Pipeline: build_vague.py -> build_questions.py -> validate_questions.py   (repeat until stable)
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_questions import VALIDATION_CACHE, load_cache  # noqa: E402
from retrievers.corpus import corpus_dir, split_frontmatter  # noqa: E402

MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.00, 10.00
MAX_ATTEMPTS = 3
CACHE = ROOT / "questions/vague_cache.jsonl"
EXCLUDED_PATH = ROOT / "questions/excluded.json"

SYSTEM = """You rewrite benchmark questions the way a real user would ask them months after reading a document: \
they remember the gist but not the exact names.

Rewrite the question so that:
- exact names of people, organizations, companies, products, programs, events and coined terms are replaced by \
descriptive references taken from the document (role, kind of organization, type of place, what it does, \
roughly when) — e.g. "the founder of the indoor-plant café" instead of "Jessica Morrison";
- it remains answerable by the document with the same answer, and is specific enough that the document is the \
only plausible match among many documents on the same topic — use the 1-2 most distinguishing details, not more;
- it does not reveal the answer;
- it sounds like something a person would actually type: one sentence, at most 30 words, in English. Referring \
to "that article/post about ..." is fine occasionally, but don't make it the default."""

SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)
_lock = threading.Lock()


def rewrite(item: dict, prev: list[dict], vcache: dict) -> tuple[str, float]:
    raw = corpus_dir("repliqa/S3000") / "raw" / item["gold_paths"][0]
    doc = split_frontmatter(raw.read_text())[1]
    feedback = ""
    for p in prev:
        v = vcache.get(p["qid"], {}).get("validation", {})
        clash = "; ".join(f"{a['path']}: {a['evidence'][:200]}" for a in v.get("answering", [])
                          if a.get("consistent_with_reference") is False)
        feedback += (f"\n\nA previous rewrite was rejected as ambiguous — it also matched other documents with a "
                     f"different answer.\nRejected: {p['question']}\nConflicting documents: {clash or 'n/a'}\n"
                     f"Add more distinguishing details from this document.")
    user = (f"<document>\n{doc}\n</document>\n\nOriginal question: {item['question_en']}\n"
            f"Reference answer: {item['answer']}{feedback}")
    resp = client.messages.create(
        model=MODEL, max_tokens=4000, system=SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"{item['qid']}: stop_reason={resp.stop_reason}")
    q = json.loads(next(b.text for b in resp.content if b.type == "text"))["question"].strip()
    return q, (resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT) / 1e6


def attempts_by_source() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if CACHE.exists():
        for r in map(json.loads, CACHE.read_text().splitlines()):
            out.setdefault(r["source_qid"], []).append(r)
    return out


if __name__ == "__main__":
    items = [json.loads(l) for l in (ROOT / "questions/main.jsonl").read_text().splitlines()]
    base = [q for q in items if q["category"] == "answerable" and q["lang"] == "en"]
    excluded = set(json.loads(EXCLUDED_PATH.read_text())) if EXCLUDED_PATH.exists() else set()
    vcache = load_cache(VALIDATION_CACHE)
    attempts = attempts_by_source()

    todo = []
    for q in base:
        prev = attempts.get(q["qid"], [])
        if not prev or (prev[-1]["qid"] in excluded and len(prev) < MAX_ATTEMPTS):
            todo.append((q, prev))
    gave_up = sum(1 for q in base if len(attempts.get(q["qid"], [])) >= MAX_ATTEMPTS
                  and attempts[q["qid"]][-1]["qid"] in excluded)

    def run(args) -> float:
        q, prev = args
        text, cost = rewrite(q, prev, vcache)
        rec = {"qid": f"v-{q['qid'][2:]}-a{len(prev) + 1}", "source_qid": q["qid"], "question": text}
        with _lock:
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return cost

    with ThreadPoolExecutor(max_workers=16) as pool:
        cost = sum(pool.map(run, todo))
    print(f"rewrote {len(todo)} (base {len(base)}, gave up after {MAX_ATTEMPTS} attempts: {gave_up}), cost ${cost:.3f}")
