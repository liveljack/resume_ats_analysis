"""Layer 5: Evaluation metrics for the no-ground-truth setting.

Three operational metrics (design.md §6):
- Spearman rank correlation (target rho > 0.8) — primary.
- ±1 absolute-score hit rate (target > 80%).
- Binary pass/fail F1 (threshold consensus with humans).

Spearman is computed without scipy (pure-python) so eval runs with no heavy deps.
"""
from __future__ import annotations

from statistics import mean


def _rank(values: list[float]) -> list[float]:
    """Average-rank of values (handles ties)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation in [-1, 1]. Pure python."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma = mean(ra)
    mb = mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = (sum((r - ma) ** 2 for r in ra)) ** 0.5
    db = (sum((r - mb) ** 2 for r in rb)) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def within_one_hit_rate(pred: list[float], gold: list[float], tol: float = 1.0) -> float:
    """Fraction of predictions within ±tol of gold."""
    if not pred:
        return 0.0
    hits = sum(1 for p, g in zip(pred, gold) if abs(p - g) <= tol)
    return hits / len(pred)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def binary_pass_f1(
    pred_scores: list[float],
    gold_scores: list[float],
    threshold: float = 6.0,
) -> dict[str, float]:
    """Pass/fail F1: scores >= threshold are 'pass'."""
    tp = fp = fn = 0
    for p, g in zip(pred_scores, gold_scores):
        pp = p >= threshold
        gg = g >= threshold
        if pp and gg:
            tp += 1
        elif pp and not gg:
            fp += 1
        elif not pp and gg:
            fn += 1
    prec, rec, f1 = _prf(tp, fp, fn)
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(pred: list[float], gold: list[float], threshold: float = 6.0, tol: float = 1.0) -> dict:
    """Run the full no-ground-truth eval suite against a human gold set."""
    return {
        "n": len(pred),
        "spearman_rho": round(spearman_rho(pred, gold), 3),
        "within_one_hit_rate": round(within_one_hit_rate(pred, gold, tol), 3),
        "binary_pass": binary_pass_f1(pred, gold, threshold),
    }
