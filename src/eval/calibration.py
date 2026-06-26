"""Calibration: fit fusion weights against a small human anchor set.

Given anchor samples with (rule_score, ml_score, llm_score, gold) tuples, solve
for the non-negative weights that minimize mean squared error between the blend
and gold, then persist them back to config via environment overrides.

Uses a tiny grid search (no scipy) — sufficient for ~50-100 anchor samples and
keeps the dependency surface minimal.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass
class AnchorSample:
    rule: float
    ml: float | None
    llm: float | None
    gold: float  # human score 1-10


def _blend(sample: AnchorSample, weights: dict[str, float]) -> float:
    sig = {"rule": sample.rule, "ml": sample.ml, "llm": sample.llm}
    avail = {k: v for k, v in sig.items() if v is not None and weights.get(k, 0) > 0}
    if not avail:
        return 0.0
    total_w = sum(weights[k] for k in avail)
    return sum(avail[k] * weights[k] for k in avail) / total_w


def fit_match_weights(samples: list[AnchorSample], step: float = 0.1) -> dict[str, float]:
    """Grid-search match fusion weights minimizing MSE vs gold."""
    grid = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    best = None
    best_err = float("inf")
    for wr, wm, wl in itertools.product(grid, grid, grid):
        if wr + wm + wl == 0:
            continue
        w = {"rule": wr, "ml": wm, "llm": wl}
        err = 0.0
        for s in samples:
            err += (_blend(s, w) - s.gold) ** 2
        if err < best_err:
            best_err = err
            best = w
    return best or {"rule": 0.2, "ml": 0.3, "llm": 0.5}


def fit_quality_weights(samples: list[AnchorSample], step: float = 0.1) -> dict[str, float]:
    """Grid-search quality fusion weights (rule + llm)."""
    grid = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    best = None
    best_err = float("inf")
    for wr, wl in itertools.product(grid, grid):
        if wr + wl == 0:
            continue
        w = {"rule": wr, "llm": wl}
        err = 0.0
        for s in samples:
            err += (_blend(s, w) - s.gold) ** 2
        if err < best_err:
            best_err = err
            best = w
    return best or {"rule": 0.3, "llm": 0.7}
