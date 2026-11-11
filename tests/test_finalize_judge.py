"""Question-set exclusions (scripts/kb2_finalize.py) and the judge prompt's use of aliases."""

import copy

from judge import prompt
from scripts.kb2_finalize import finalize


def q(qid, cat, scales=(300, 900, 3000), **kw):
    return {"qid": qid, "category": cat, "scales": list(scales), "answer": kw.pop("answer", "X"), **kw}


def run(qs, val=None, bscale=None, leak2=None, manual=None, gold=None):
    return {x["qid"]: x for x in finalize(copy.deepcopy(qs), val or {}, bscale or {}, leak2 or {}, manual or {},
                                          gold or {})}


def test_clean_question_untouched():
    r = run([q("a", "musique_ans")])["a"]
    assert not r["excluded"] and r["valid_scales"] == [300, 900, 3000] and "exclude_reason" not in r


def test_first_leak_check_excludes_everywhere():
    r = run([q("u", "musique_unans")],
            val={"u": {"answerable_from_candidates": True, "supporting_paths": ["p"], "evidence": "e"}})["u"]
    assert r["excluded"] and "check 1" in r["exclude_reason"]


def test_second_leak_check_is_per_scale():
    r = run([q("u", "musique_unans")],
            leak2={"u": {"leaked_scales": [3000], "supporting_paths": ["p"], "evidence": "e"}})["u"]
    assert not r["excluded"] and r["valid_scales"] == [300, 900] and "check 2" in r["exclude_reason"]


def test_leak_at_every_asked_scale_excludes():
    r = run([q("u", "bright_unans", scales=(900, 3000))],
            bscale={("u", 900): {"substantive_answer_possible": True}},
            leak2={"u": {"leaked_scales": [3000], "sufficient_paths": ["p"], "evidence": "e"}})["u"]
    assert r["excluded"] and r["valid_scales"] == []


def test_manual_leak_and_gold_drop():
    out = run([q("u", "musique_unans"), q("a", "musique_ans")],
              manual={"u": {"reason": "hand"}},
              gold={"a": {"action": "drop", "defect": "wrong_referent", "reason": "r", "aliases": []}})
    assert out["u"]["excluded"] and "hand review" in out["u"]["exclude_reason"]
    assert out["a"]["excluded"] and "wrong_referent" in out["a"]["exclude_reason"]


def test_aliases_added_once_and_idempotent():
    gold = {"a": {"action": "add_aliases", "defect": "none", "reason": "r", "aliases": ["1642", "X", "Dec 1642"]}}
    qs = [q("a", "musique_ans", answer="X", answer_aliases=["Dec 1642"])]
    once = run(qs, gold=gold)["a"]
    assert once["answer_aliases"] == ["Dec 1642", "1642"]  # no duplicate, reference itself not added
    twice = run([once], gold=gold)["a"]
    assert twice["answer_aliases"] == once["answer_aliases"] and not twice["excluded"]


def test_judge_prompt_lists_aliases():
    question = {"category": "musique_ans", "answerable": True, "question": "When?", "answer": "13 December 1642",
                "answer_aliases": ["1642", "December 1642"]}
    p = prompt({"answer": "In 1642."}, question)
    assert "Reference answer: 13 December 1642" in p and "Aliases: 1642, December 1642" in p
    unans = prompt({"answer": "x"}, {"category": "musique_unans", "answerable": False, "question": "Q"})
    assert "Reference answer: none" in unans
