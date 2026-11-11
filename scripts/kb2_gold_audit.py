"""v2: audit the gold labels of MuSiQue answerable questions (blind to every model's answers).

MuSiQue composes single-hop questions into one multi-hop question. The composition can go wrong (a description
that points at a different entity, a false premise), and the reference answer can be wrong or only one of
several correct answers. The original v2 pipeline never checked this. Claude Opus 5.5 sees the question, the
reference + aliases, the MuSiQue decomposition and the full gold documents — never a model response, so the
audit cannot favour any model under test — and returns a verdict, judged relative to the documents (the
assistant under test must take its facts from the knowledge base):
  ok           reference correct and the only answer the documents support
  add_aliases  question fine, but the documents support other correct answers (other_valid_answers)
  drop         a named defect: broken_chain, wrong_referent, wrong_answer_type, false_premise, wrong_reference

A first pass judged against world knowledge instead (questions_v2/gold_audit_strict.jsonl, 92 of 200 "drop"):
it dropped questions for time-dependence and for answers known only outside the knowledge base, which the
benchmark's answering rules don't allow anyway. Kept for comparison.

Results -> questions_v2/gold_audit.jsonl (per qid, append-as-you-go, resumable). Decisions are applied by
kb2_finalize.py from questions_v2/gold_decisions.json after a human review of every non-ok verdict.

  uv run python -I scripts/kb2_gold_audit.py [--limit N]
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

from retrievers.corpus import corpus_dir, split_frontmatter  # noqa: E402

MODEL, PRICE = "claude-opus-5-5", (4.00, 20.00)
CORPUS = "kb2/S3000"  # every scale is a subset of S3000; gold docs of every question are in it
MUSIQUE = ROOT / "data/raw/musique/musique_full_v1.0_dev.jsonl"
OUT = ROOT / "questions_v2/gold_audit.jsonl"

SYSTEM = """You audit the gold labels of a multi-hop question-answering benchmark built from MuSiQue. MuSiQue \
composes single-hop questions into one multi-hop question by replacing a bridge entity with a description of \
it. That composition sometimes goes wrong: a description can point at a different entity than intended, a \
premise can be false, or the reference answer can be wrong or answer something other than what is asked.

The benchmark tests a knowledge-base assistant. It must take its facts from the knowledge base (the documents \
below are the relevant part of it) and may use general knowledge only to interpret and connect them — for \
example which city a landmark is in, or what a term means. So judge every question relative to the documents: \
the documents are the ground truth at their snapshot.

You see the multi-hop question, its reference answer and aliases, the decomposition (single-hop questions with \
their intended answers) and the full text of the supporting documents. Read the question as a careful reader \
would read it as worded, not as the composition intended.

verdict "drop" only for a real defect, i.e. one of:
- broken_chain: a hop's intended answer is neither in the documents nor an obvious connection of the kind the \
rules allow, so a careful reader of the documents cannot reach the reference.
- wrong_referent: read with the documents, a description in the question picks out a different entity than \
intended (e.g. a superlative the documents attribute to someone else, a place the documents put elsewhere, a \
parent asked for where the reference is a child, a fictional character conflated with a real person).
- wrong_answer_type: the reference does not answer what is asked (a release date for a writing date, a day for \
a year, the year of a cause for the year of the event).
- false_premise: the question presupposes something the documents contradict.
- wrong_reference: the documents themselves give a different answer to the question as worded.
Do NOT drop for: time-dependence when the documents state the value (the knowledge base snapshot is the truth); \
other answers known only from outside the documents; a referent that is ambiguous in the world but that the \
documents resolve one way; a defect in a hop that a careful reader of the documents still gets through to the \
reference.

verdict "add_aliases" when the question is fine but the documents also support other correct answers to it as \
worded (distinct entities or values, not paraphrases, translations or formatting variants). List them in \
other_valid_answers; also list a less specific form (such as the year of a full date) when the question's \
wording makes it a complete answer, and well-known members of a small closed set the question admits (e.g. the \
other wars a navy served in). Otherwise verdict "ok".

Return defect (one of the five names above, or "none"), verdict, correct_answer (if wrong_reference or \
wrong_answer_type: the answer the documents give, else empty), other_valid_answers, and reason (one to three \
sentences quoting the document text that decides it). Be strict about real defects and do not flag pedantry."""

SCHEMA = {
    "type": "object",
    "properties": {
        "defect": {"type": "string", "enum": ["none", "broken_chain", "wrong_referent", "wrong_answer_type",
                                              "false_premise", "wrong_reference"]},
        "verdict": {"type": "string", "enum": ["ok", "add_aliases", "drop"]},
        "correct_answer": {"type": "string"},
        "other_valid_answers": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["defect", "verdict", "correct_answer", "other_valid_answers", "reason"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=6)
lock = threading.Lock()


def decomposition(row: dict) -> str:
    lines = []
    for i, h in enumerate(row["question_decomposition"], 1):
        lines.append(f"{i}. {h['question']}  ->  {h['answer']}")
    return "\n".join(lines) + "\n(#k in a hop refers to the answer of hop k)"


def user_prompt(q: dict, row: dict) -> str:
    raw = corpus_dir(CORPUS) / "raw"
    docs = "\n\n".join(f'<document path="{p}">\n{split_frontmatter((raw / p).read_text())[1]}\n</document>'
                       for p in q["gold_paths"])
    aliases = ", ".join(q.get("answer_aliases") or []) or "(none)"
    return (f"{docs}\n\nMulti-hop question: {q['question_en']}\nReference answer: {q['answer']}\n"
            f"Aliases: {aliases}\n\nDecomposition:\n{decomposition(row)}")


def audit(q: dict, row: dict) -> tuple[dict, float]:
    r = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user_prompt(q, row)}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"{q['qid']}: stop_reason={r.stop_reason}")
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    v["audit_model"] = r.model
    return v, (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    qs = [q for q in map(json.loads, open(ROOT / "questions_v2/questions.jsonl")) if q["category"] == "musique_ans"]
    want = {q["src_id"] for q in qs}
    rows = {r["id"]: r for r in map(json.loads, open(MUSIQUE)) if r["id"] in want}
    done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [q for q in qs if q["qid"] not in done][:limit]

    def run(q: dict) -> float:
        v, c = audit(q, rows[q["src_id"]])
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": q["qid"], "question_en": q["question_en"], "answer": q["answer"],
                                    **v, "audit_cost": c}, ensure_ascii=False) + "\n")
        return c

    with ThreadPoolExecutor(8) as pool:
        cost = sum(pool.map(run, todo))
    res = [json.loads(l) for l in OUT.read_text().splitlines()]
    from collections import Counter
    print(f"audited {len(todo)} (cached {len(done)}), cost ${cost:.2f}")
    print("verdicts:", dict(Counter(r["verdict"] for r in res)))
