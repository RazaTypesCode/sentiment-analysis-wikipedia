"""
Statistical comparison of sentiment scores between two article groups.

Uses:
  - Welch's t-test     (does not assume equal variance)
  - Mann-Whitney U     (non-parametric, robust to small / non-normal samples)

Both tests use a two-sided alternative hypothesis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ── Summary ────────────────────────────────────────────────────────────────────

def summarize(scores: list[float], name: str = "") -> dict:
    arr = np.array(scores, dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return {
        "name": name,
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "median": round(float(np.median(arr)), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare_groups(
    scores_a: list[float],
    scores_b: list[float],
    name_a: str = "Group A",
    name_b: str = "Group B",
) -> dict:
    """
    Returns a result dict with summary statistics and two test results.
    Both tests use the two-sided alternative.
    """
    t_stat, t_p = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")

    return {
        "group_a": summarize(scores_a, name_a),
        "group_b": summarize(scores_b, name_b),
        "welch_t": {
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(t_p), 6),
        },
        "mann_whitney_u": {
            "statistic": round(float(u_stat), 2),
            "p_value": round(float(u_p), 6),
        },
        "significant_05": bool(t_p < 0.05 or u_p < 0.05),
        "significant_01": bool(t_p < 0.01 and u_p < 0.01),
    }


def print_comparison(result: dict) -> None:
    a, b = result["group_a"], result["group_b"]
    w = 54
    print(f"\n{'─' * w}")
    print(f"  {'Group':<20}  {'n':>3}   {'mean':>7}   {'std':>6}   {'median':>7}")
    print(f"  {'─'*20}  {'─'*3}   {'─'*7}   {'─'*6}   {'─'*7}")
    print(f"  {a['name']:<20}  {a['n']:>3}   {a['mean']:>+7.3f}   {a['std']:>6.3f}   {a['median']:>+7.3f}")
    print(f"  {b['name']:<20}  {b['n']:>3}   {b['mean']:>+7.3f}   {b['std']:>6.3f}   {b['median']:>+7.3f}")
    print(f"{'─' * w}")
    t, u = result["welch_t"], result["mann_whitney_u"]
    print(f"  Welch t-test    :  t = {t['statistic']:+.3f}   p = {t['p_value']:.4f}")
    print(f"  Mann-Whitney U  :  U = {u['statistic']:.1f}    p = {u['p_value']:.4f}")
    print(f"{'─' * w}")
    if result["significant_01"]:
        verdict = "SIGNIFICANT at p < 0.01  ★★"
    elif result["significant_05"]:
        verdict = "SIGNIFICANT at p < 0.05  ★"
    else:
        verdict = "not significant"
    print(f"  Result  :  {verdict}")
    print(f"{'─' * w}\n")


# ── Multi-group helper ─────────────────────────────────────────────────────────

def compare_many(
    groups: dict[str, list[float]],
) -> dict[tuple[str, str], dict]:
    """Compare all pairs in a dict of {name: scores}. Returns pairwise results."""
    names = list(groups.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            results[(a, b)] = compare_groups(groups[a], groups[b], a, b)
    return results
