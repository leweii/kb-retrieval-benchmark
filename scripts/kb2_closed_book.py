"""v2 step A: closed-book filter for MuSiQue candidates with Claude Sonnet 5.5.

Candidates = answerable MuSiQue dev questions in a fixed hash order. Each is asked without any knowledge base;
questions Sonnet answers correctly are excluded from the v2 question set (they'd measure memory, not retrieval).
Cached per question id in questions_v2/closed_book_sonnet.jsonl (append-as-you-go, resumable).
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from closed_book import f1, norm  # noqa: E402

MODEL, PRICE = "claude-sonnet-5-5", (2.00, 10.00)
N_CANDIDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 420
OUT = ROOT / "questions_v2/closed_book_sonnet.jsonl"
SYSTEM = ("Answer the question from your own knowledge with a short phrase only (a name, date, number or place). "
          "If you don't know, answer exactly: unknown")

client = anthropic.Anthropic(max_retries=5)
lock = threading.Lock()


def hkey(s: str) -> str:
    return hashlib.sha1(f"musique:{s}".encode()).hexdigest()


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(ROOT / "data/raw/musique/musique_full_v1.0_dev.jsonl")]
    cands = sorted((r for r in rows if r["answerable"]), key=lambda r: hkey(r["id"]))[:N_CANDIDATES]
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [r for r in cands if r["id"] not in done]

    def one(r) -> float:
        resp = client.messages.create(model=MODEL, max_tokens=3000, system=SYSTEM, output_config={"effort": "low"},
                                      messages=[{"role": "user", "content": r["question"]}])
        pred = "".join(b.text for b in resp.content if b.type == "text").strip()
        golds = [r["answer"]] + r.get("answer_aliases", [])
        correct = any(norm(g) and (norm(g) in norm(pred) or f1(pred, g) >= 0.8) for g in golds)
        rec = {"id": r["id"], "pred": pred, "gold": r["answer"], "correct": correct}
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return (resp.usage.input_tokens * PRICE[0] + resp.usage.output_tokens * PRICE[1]) / 1e6

    with ThreadPoolExecutor(16) as pool:
        cost = sum(pool.map(one, todo))
    recs = [json.loads(l) for l in OUT.read_text().splitlines()]
    print(f"asked {len(todo)} (cached {len(done)}), cost ${cost:.2f} | closed-book correct "
          f"{sum(r['correct'] for r in recs)}/{len(recs)} -> usable {sum(not r['correct'] for r in recs)}")
