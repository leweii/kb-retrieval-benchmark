"""Significance tests and robustness checks for the report, computed from the judged records.

  mcnemar_exact      exact two-sided McNemar test on paired outcomes (same questions, two cells)
  holm               Holm–Bonferroni adjustment over a family of p-values
  question_clusters  MuSiQue questions that share a gold document or an answer form one cluster (they are not
                     independent: e.g. five questions built on the same Nanjing page)
  cluster_bootstrap  95% interval of an accuracy difference, resampling clusters instead of questions
  significance       the report's key comparisons at every scale, with Holm-adjusted p and cluster intervals
  threshold_sensitivity  recommendation picks under different correct-refusal thresholds

Everything is computed on the MuSiQue clean set with the report's scoring rule (build_report.Data.cell): a run
without a usable response counts as wrong.
"""

from __future__ import annotations

import random
import re
from math import comb

COMPARISONS = [  # (model, mode, config) vs (model, mode, config): the claims the report makes
    (("deepseek", "A1", "R1_grep"), ("sonnet", "A1", "R1_grep")),
    (("deepseek", "A1", "R1_grep"), ("haiku", "A1", "R1_grep")),
    (("deepseek", "A1", "R1_grep"), ("deepseek", "A1", "R5_wiki_nav")),
    (("deepseek", "A1", "R1_grep"), ("deepseek", "A1", "R3_hybrid")),
    (("deepseek", "A1", "R1_grep"), ("deepseek", "A2", "R3_hybrid")),
    (("deepseek", "A1", "R1_grep"), ("deepseek", "A0", "R4_enhanced")),
    (("sonnet", "A1", "R1_grep"), ("sonnet", "A1", "R3_hybrid")),
]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c = discordant pairs (A right & B wrong, A wrong & B right)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals: dict) -> dict:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1), same keys as the input."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def _norm(s) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(s).lower()).split())


def question_clusters(qs: dict) -> dict[str, str]:
    """qid -> cluster id for MuSiQue answerable questions: union of questions sharing a gold path or an answer."""
    ids = [q for q, v in qs.items() if v["category"] == "musique_ans"]
    parent = {q: q for q in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for q in ids:
        for key in [f"doc:{p}" for p in qs[q]["gold_paths"]] + [f"ans:{_norm(qs[q]['answer'])}"]:
            if key in owner:
                parent[find(q)] = find(owner[key])
            else:
                owner[key] = q
    return {q: find(q) for q in ids}


def cluster_bootstrap(pairs: dict[str, tuple[bool, bool]], cluster_of: dict[str, str], n: int = 10000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Accuracy difference (A - B) on paired questions and its 95% percentile interval, resampling clusters."""
    if not pairs:
        return (0.0, 0.0, 0.0)
    groups: dict[str, list[tuple[bool, bool]]] = {}
    for q, ab in pairs.items():
        groups.setdefault(cluster_of.get(q, q), []).append(ab)
    keys = sorted(groups)
    diff = sum(a - b for a, b in pairs.values()) / len(pairs)
    rng, sims = random.Random(seed), []
    for _ in range(n):
        tot = cnt = 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            tot += sum(a - b for a, b in g)
            cnt += len(g)
        sims.append(tot / cnt)
    sims.sort()
    return (diff, sims[int(0.025 * (n - 1))], sims[int(0.975 * (n - 1))])


def paired(d, a: tuple, b: tuple, scale: int) -> dict[str, tuple[bool, bool]]:
    """qid -> (A correct, B correct) on clean MuSiQue questions judged in both cells (no response = wrong)."""
    def outcomes(m, mode, cfg):
        return {r["qid"]: bool(r["judge"]["correct"]) for r in d.g_jud[(m, mode, cfg, scale)]
                if r["category"] == "musique_ans" and r["qid"] not in d.memorized}
    x, y = outcomes(*a), outcomes(*b)
    return {q: (x[q], y[q]) for q in x.keys() & y.keys()}


def significance(d, scales=(300, 900, 3000), comparisons=COMPARISONS, n_boot: int = 10000) -> list[dict]:
    """One row per (scale, comparison): accuracies, discordant counts, McNemar p, Holm-adjusted p over the whole
    family, and the cluster-bootstrap interval of the difference."""
    clusters = question_clusters(d.qs)
    rows = []
    for s in scales:
        for a, b in comparisons:
            p = paired(d, a, b, s)
            bb = sum(x and not y for x, y in p.values())
            cc = sum(y and not x for x, y in p.values())
            diff, lo, hi = cluster_bootstrap(p, clusters, n_boot)
            rows.append({"scale": s, "a": a, "b": b, "n": len(p),
                         "acc_a": sum(x for x, _ in p.values()) / len(p) if p else None,
                         "acc_b": sum(y for _, y in p.values()) / len(p) if p else None,
                         "b_only": bb, "c_only": cc, "p": mcnemar_exact(bb, cc),
                         "n_clusters": len({clusters.get(q, q) for q in p}), "diff": diff, "lo": lo, "hi": hi})
    adj = holm({i: r["p"] for i, r in enumerate(rows)})
    for i, r in enumerate(rows):
        r["p_holm"] = adj[i]
    return rows


def threshold_sensitivity(d, recommend, scales=(300, 900, 3000), thresholds=(0.7, 0.8)) -> dict:
    """{scale: {threshold: (model, label, acc, refusal_ok) of the quality pick or None}}."""
    out = {}
    for s in scales:
        out[s] = {}
        for t in thresholds:
            r = recommend(d, s, min_refusal=t).get("best")
            out[s][t] = (r[0], r[1], r[2]["acc"], r[2]["refusal_ok"]) if r else None
    return out
