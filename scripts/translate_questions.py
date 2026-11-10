"""Fill `question` for lang=zh items by translating question_en with Claude Sonnet 5.5.

Proper nouns (people, organizations, products, places, publications) must stay in English — mirrors how a
Chinese speaker asks about an English knowledge base, and keeps grep/BM25 comparable. Every translation is
validated (all English proper-noun tokens present verbatim, no meta chatter) and retried on failure.

Usage: translate_questions.py [--force] [files...]
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.00, 10.00  # $/M tokens
MAX_ATTEMPTS = 4

SYSTEM = """You translate English questions into natural, fluent Simplified Chinese, phrased the way a \
Chinese-speaking user would type the question into a knowledge-base search box.

Rules:
- Keep every proper noun exactly as written in English: names of people, organizations, companies, \
products, publications/media outlets, events, programs and places. Do not transliterate them \
(write "Sarah Johnson", not "莎拉·约翰逊"; "Walmart", not "沃尔玛").
- Keep numbers and dates exact.
- The input may be a long forum post: translate all of it and keep its paragraph structure. Leave code, \
commands, file names, URLs, error messages and identifiers unchanged.
- Translate the question only; never answer it or add commentary."""

SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}

# Capitalized words that are sentence starters / function words / calendar words, not proper nouns.
NOT_PROPER = set("""What How Who Whom Whose Which Why When Where Does Did Do Is Are Was Were Has Have Had Can Could
Would Should Will According Considering Based Given Between Both After Before Following Regarding In On Of For And
Or To By From With As At If The A An This That These Those It Its Their His Her Our Your During Since Among Under
Over Within Without Into Upon Also Not No Yes Than Then There Here Such Each Every Other Some Any All Which
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December""".split())

client = anthropic.Anthropic(max_retries=5)


def proper_tokens(text: str) -> set[str]:
    """Capitalized tokens that are likely proper nouns. Sentence-initial capitals are ignored unless the token
    has internal capitals or digits (ROS1, McKenna) — otherwise long posts flag every 'Running', 'Thanks'."""
    out = set()
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9&'’\-]*[A-Za-z0-9]", text):
        t = m.group(0)
        if t in NOT_PROPER:
            continue
        before = text[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:\n\"“(*#-"
        if sentence_start and not re.search(r"[A-Z0-9]", t[1:]):
            continue
        out.add(t)
    return out


def problems(src: str, zh: str) -> list[str]:
    issues = []
    missing = [t for t in proper_tokens(src) if t.rstrip("’'s") not in zh and t not in zh]
    if missing:
        issues.append(f"missing proper nouns: {missing[:12]}")
    if ("\n" in zh and "\n" not in src) or re.search(r"(?i)\btranslat|\bwait\b|原文", zh):
        issues.append("meta chatter")
    if not re.search(r"[一-鿿]", zh):
        issues.append("no Chinese")
    return issues


def translate(text: str) -> tuple[str, float, list[str]]:
    cost, feedback = 0.0, ""
    for _ in range(MAX_ATTEMPTS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": text + feedback}],
        )
        cost += (resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT) / 1e6
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={resp.stop_reason} for: {text[:80]}")
        zh = json.loads(next(b.text for b in resp.content if b.type == "text"))["translation"].strip()
        issues = problems(text, zh)
        if not issues:
            return zh, cost, []
        feedback = f"\n\n(Previous attempt was rejected — {'; '.join(issues)}. Keep those words in English.)"
    return zh, cost, issues


CACHE = ROOT / "questions/translation_cache.jsonl"
_lock = threading.Lock()


def process(path: Path, force: bool) -> float:
    items = [json.loads(l) for l in path.read_text().splitlines()]
    cache = {r["qid"]: r for r in map(json.loads, CACHE.read_text().splitlines())} if CACHE.exists() else {}
    todo = [q for q in items if q["lang"] == "zh"
            and (force or q["qid"] not in cache or cache[q["qid"]]["question_en"] != q["question_en"])]

    def run(q: dict) -> float:
        zh, c, issues = translate(q["question_en"])
        rec = {"qid": q["qid"], "question_en": q["question_en"], "question": zh, "translation_issues": issues}
        with _lock:  # append as soon as each finishes: an interrupted run loses nothing
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[q["qid"]] = rec
        return c

    with ThreadPoolExecutor(max_workers=16) as pool:
        cost = sum(pool.map(run, todo))
    flagged = 0
    for q in items:
        if q["lang"] != "zh":
            q["question"] = q["question_en"]
        if q["lang"] == "zh":
            rec = cache[q["qid"]]
            q["question"] = rec["question"]
            q.pop("translation_issues", None)
            if rec["translation_issues"]:
                q["translation_issues"] = rec["translation_issues"]
                flagged += 1
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in items), encoding="utf-8")
    print(f"{path.name}: translated {len(todo)}, still flagged {flagged}, cost ${cost:.3f}")
    return cost


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    files = [Path(a) for a in args if a != "--force"] or [ROOT / "questions/main.jsonl", ROOT / "questions/side.jsonl"]
    total = sum(process(f, force) for f in files)
    print(f"total ${total:.3f}")
