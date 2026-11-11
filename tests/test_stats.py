import pytest

from stats import cluster_bootstrap, holm, mcnemar_exact, question_clusters


def test_mcnemar_matches_known_values():
    # b=19, c=6: the pre-audit DeepSeek vs Sonnet comparison the report quoted as p = 0.015
    assert mcnemar_exact(19, 6) == pytest.approx(0.01463, abs=1e-4)
    assert mcnemar_exact(1, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)


def test_mcnemar_is_symmetric_and_bounded():
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0


def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted: a 0.01*3, c 0.03*2, b 0.04*1 -> monotone: 0.03, 0.06, 0.06
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.9, "y": 0.8}) == {"x": 1.0, "y": 1.0}


def q(qid, paths, answer, cat="musique_ans"):
    return qid, {"category": cat, "gold_paths": paths, "answer": answer}


def test_clusters_join_shared_documents_and_answers():
    qs = dict([q("a", ["wiki/x.md"], "Paris"), q("b", ["wiki/x.md", "wiki/y.md"], "Rome"),
               q("c", ["wiki/z.md"], "rome."), q("d", ["wiki/w.md"], "Oslo"),
               q("e", ["wiki/x.md"], "Paris", cat="musique_unans")])
    cl = question_clusters(qs)
    assert cl["a"] == cl["b"] == cl["c"]  # a-b share a document, b-c share a (normalized) answer
    assert cl["d"] != cl["a"]
    assert "e" not in cl  # only answerable MuSiQue questions


def test_cluster_bootstrap_identical_arms_gives_zero():
    pairs = {str(i): (i % 2 == 0, i % 2 == 0) for i in range(20)}
    assert cluster_bootstrap(pairs, {}, n=200) == (0.0, 0.0, 0.0)


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    pairs = {str(i): (True, i % 3 == 0) for i in range(30)}
    r1 = cluster_bootstrap(pairs, {str(i): str(i // 3) for i in range(30)}, n=500, seed=1)
    r2 = cluster_bootstrap(pairs, {str(i): str(i // 3) for i in range(30)}, n=500, seed=1)
    assert r1 == r2
    diff, lo, hi = r1
    assert diff == pytest.approx(20 / 30)
    assert lo <= diff <= hi


def test_cluster_bootstrap_empty():
    assert cluster_bootstrap({}, {}) == (0.0, 0.0, 0.0)
