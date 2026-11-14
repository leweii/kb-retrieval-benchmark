"""Scoring rules of the report: cell() denominators, evidence measures, recommend() thresholds and ties."""

from collections import defaultdict

import pytest

from build_report import Data, evidence_found, evidence_noise, recommend


def rec(qid, cat, correct=None, abstained=False, retrieved=(), read=(), cost=0.01, steps=3):
    return {"qid": qid, "category": cat, "lang": "en", "retrieved": list(retrieved), "read": list(read),
            "read_raw": [], "latency": 1.0, "cost": cost, "steps": steps, "tool_calls": 1,
            "judge": {"correct": correct, "abstained": abstained}}


def make_data(qs: dict, cells: dict, memorized=()) -> Data:
    """A Data without files: qs = {qid: question}, cells = {(model, mode, config, scale): [records]}."""
    d = Data.__new__(Data)
    d.qs, d.memorized = qs, set(memorized)
    d.g_raw, d.g_jud = defaultdict(list), defaultdict(list)
    for k, rs in cells.items():
        d.g_raw[k] = rs
        d.g_jud[k] = rs
    return d


def mq(qid, answerable=True, cat="musique_ans", gold=("wiki/a.md", "wiki/b.md")):
    return {"qid": qid, "category": cat, "answerable": answerable, "gold_paths": list(gold)}


K = ("deepseek", "A1", "R1_grep", 3000)


def test_no_response_counts_as_wrong_in_accuracy():
    qs = {f"q{i}": mq(f"q{i}") for i in range(4)}
    rs = [rec("q0", "musique_ans", True), rec("q1", "musique_ans", True), rec("q2", "musique_ans", False),
          rec("q3", "musique_ans", None, abstained=None)]  # no response: judge marks both null
    c = make_data(qs, {K: rs}).cell(*K)
    assert c["acc_n"] == 4 and c["acc"] == pytest.approx(0.5)
    assert c["no_response"] == 1


def test_no_response_is_a_non_answer_and_not_a_correct_refusal():
    qs = {"a": mq("a"), "u1": mq("u1", False, "musique_unans"), "u2": mq("u2", False, "musique_unans")}
    rs = [rec("a", "musique_ans", None, abstained=None),
          rec("u1", "musique_unans", None, abstained=True), rec("u2", "musique_unans", None, abstained=None)]
    c = make_data(qs, {K: rs}).cell(*K)
    assert c["false_ref"] == 1.0  # the answerable question got no answer
    assert c["refusal_ok"] == pytest.approx(0.5) and c["refusal_n"] == 2


def test_memorized_questions_are_left_out_of_accuracy_only():
    qs = {"m": mq("m"), "x": mq("x")}
    rs = [rec("m", "musique_ans", True), rec("x", "musique_ans", False)]
    c = make_data(qs, {K: rs}, memorized={"m"}).cell(*K)
    assert c["acc_n"] == 1 and c["acc"] == 0.0
    assert c["false_ref"] == 0.0  # memorized question still counts for refusal behaviour


def test_evidence_found_exposure_vs_opened():
    r = rec("q", "musique_ans", retrieved=["wiki/a.md", "wiki/b.md"], read=["wiki/a.md"])
    assert evidence_found(r, ["wiki/a.md", "wiki/b.md"])  # shown: grep line / snippet counts
    assert not evidence_found(r, ["wiki/a.md", "wiki/b.md"], opened_only=True)
    assert evidence_found(r, ["wiki/a.md", "wiki/b.md"], all_=False, opened_only=True)


def test_evidence_noise_is_share_of_shown_docs_that_are_not_evidence():
    gold = ["wiki/a.md", "wiki/b.md"]
    r = rec("q", "musique_ans", retrieved=["wiki/a.md", "wiki/x.md", "wiki/y.md"], read=["wiki/b.md", "wiki/a.md"])
    assert evidence_noise(r, gold) == 0.5  # 4 distinct shown, 2 of them evidence
    assert evidence_noise(rec("q", "musique_ans", retrieved=gold), gold) == 0.0
    assert evidence_noise(rec("q", "musique_ans"), gold) is None  # nothing shown: no noise to measure


def test_noise_cell_skips_questions_with_nothing_shown():
    qs = {"q": mq("q"), "r": mq("r")}
    k = ("haiku", "A1", "R1_grep", 300)
    rs = [rec("q", "musique_ans", True, retrieved=["wiki/a.md", "wiki/x.md"]), rec("r", "musique_ans", False)]
    c = make_data(qs, {k: rs}).cell(*k)
    assert c["noise"] == 0.5


def test_mus_opened_is_none_for_single_shot():
    qs = {"q": mq("q")}
    k0 = ("haiku", "A0", "R2_bm25", 300)
    c = make_data(qs, {k0: [rec("q", "musique_ans", True, retrieved=["wiki/a.md", "wiki/b.md"])]}).cell(*k0)
    assert c["mus_all"] == 1.0 and c["mus_opened"] is None


# ---------------------------------------------------------------- recommend()

def cell_records(prefix, n_ans, n_right, n_unans, n_refused, cost=0.01):
    rs = [rec(f"{prefix}a{i}", "musique_ans", i < n_right, cost=cost) for i in range(n_ans)]
    rs += [rec(f"{prefix}u{i}", "musique_unans", None, abstained=i < n_refused, cost=cost) for i in range(n_unans)]
    return rs


def setup(cells_spec: dict, scale=3000):
    """cells_spec: {(model, mode, config): (n_right_of_40, n_refused_of_10, cost)} -> Data (shared question ids)."""
    qs = {}
    for i in range(40):
        qs[f"a{i}"] = mq(f"a{i}")
    for i in range(10):
        qs[f"u{i}"] = mq(f"u{i}", False, "musique_unans")
    cells = {(*k, scale): cell_records("", 40, right, 10, refused, cost) for k, (right, refused, cost) in
             cells_spec.items()}
    return make_data(qs, cells)


def test_recommend_threshold_excludes_low_refusal():
    d = setup({("deepseek", "A1", "R1_grep"): (38, 6, 0.01),   # 95% acc, 60% refusal -> excluded at 0.7
               ("sonnet", "A1", "R1_grep"): (32, 9, 0.02)})    # 80% acc, 90% refusal
    assert recommend(d, 3000)["best"][0] == "sonnet"
    assert recommend(d, 3000, min_refusal=0.5)["best"][0] == "deepseek"


def test_recommend_requires_min_questions():
    d = setup({("deepseek", "A1", "R1_grep"): (38, 9, 0.01)})
    assert recommend(d, 3000, min_n=41) == {}
    assert recommend(d, 3000, min_n=40)["best"][0] == "deepseek"


def test_recommend_ties_break_on_refusal_then_cost():
    d = setup({("haiku", "A1", "R1_grep"): (36, 8, 0.001),
               ("sonnet", "A1", "R1_grep"): (36, 9, 0.05),
               ("deepseek", "A1", "R1_grep"): (36, 9, 0.004)})
    r = recommend(d, 3000)
    assert r["best"][0] == "deepseek"  # same accuracy; higher refusal beats haiku, lower cost beats sonnet
    assert r["value"][0] == "haiku"  # cheapest within 5 points


def test_recommend_value_and_flash_respect_margin():
    d = setup({("sonnet", "A1", "R1_grep"): (38, 9, 0.05),      # 95%
               ("deepseek", "A1", "R1_grep"): (36, 9, 0.004),   # 90%: 5 points below -> within margin
               ("haiku", "A1", "R1_grep"): (35, 9, 0.001)})     # 87.5%: outside the margin of 95%
    r = recommend(d, 3000)
    assert r["best"][0] == "sonnet" and r["value"][0] == "deepseek"
    assert r["flash"][0] == "haiku"  # among haiku / deepseek: best 90%, haiku within 5 points and cheaper
    assert r["worst"][0] == "haiku"
