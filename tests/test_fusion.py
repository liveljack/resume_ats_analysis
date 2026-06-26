"""Tests for Layer 4 (fusion) & Layer 5 (eval)."""
import pytest

from src.eval.calibration import AnchorSample, fit_match_weights, fit_quality_weights
from src.eval.metrics import (
    binary_pass_f1,
    spearman_rho,
    within_one_hit_rate,
)
from src.fusion.fusion import fuse_match, fuse_quality


def test_fuse_quality_blends_rule_and_llm():
    r = fuse_quality(rule_score=6.0, rule_dims=[], llm_score=8.0)
    # weights 0.3/0.7 => 0.3*6 + 0.7*8 = 7.4
    assert abs(r.score - 7.4) < 0.05
    assert r.llm_score == 8.0


def test_fuse_quality_redistributes_when_no_llm():
    # No LLM => all weight on rule.
    r = fuse_quality(rule_score=6.0, rule_dims=[], llm_score=None)
    assert abs(r.score - 6.0) < 0.05
    assert r.llm_score is None


def test_fuse_match_redistributes_when_ml_missing():
    r = fuse_match(rule_score=6.0, ml_score=None, llm_score=8.0)
    # weights rule0.2 ml0.3 llm0.5; ml missing => redistribute 0.3 over rule+llm
    # available weights normalized: rule 0.2/(0.7)=0.2857, llm 0.5/0.7=0.7143
    expected = 6.0 * (0.2 / 0.7) + 8.0 * (0.5 / 0.7)
    assert abs(r.score - expected) < 0.05


def test_fuse_match_all_missing_returns_floor():
    # rule=0.0 is an available (zero) signal; blend 0 maps to the 1-10 floor.
    r = fuse_match(rule_score=0.0, ml_score=None, llm_score=None)
    assert r.score == 1.0


def test_spearman_perfect_correlation():
    a = [1, 2, 3, 4, 5]
    assert abs(spearman_rho(a, [2, 4, 6, 8, 10]) - 1.0) < 1e-6


def test_spearman_inverse():
    a = [1, 2, 3, 4, 5]
    assert spearman_rho(a, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_within_one_hit_rate():
    pred = [7, 8, 3, 9]
    gold = [7, 7, 5, 9]
    # within tol=1: 7vs7(0) yes, 8vs7(1) yes, 3vs5(2) no, 9vs9(0) yes => 3/4
    assert within_one_hit_rate(pred, gold, tol=1.0) == 0.75
    # tol=0.5: 7vs7 yes, 8vs7 no, 3vs5 no, 9vs9 yes => 2/4
    assert within_one_hit_rate(pred, gold, tol=0.5) == 0.5


def test_binary_pass_f1():
    pred = [7, 8, 3, 9, 5]
    gold = [6, 8, 4, 9, 6]  # threshold 6: gold pass = idx0,1,3,4
    res = binary_pass_f1(pred, gold, threshold=6.0)
    # pred pass = idx0,1,3; gold pass = idx0,1,3,4
    # TP=3 (0,1,3), FP=0, FN=1 (idx4 gold-pass pred-fail)
    assert res["tp"] == 3
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert 0 < res["f1"] <= 1.0


def test_calibration_fits_weights():
    # Gold tracks LLM strongly; fit should weight llm higher than rule.
    samples = [
        AnchorSample(rule=5.0, ml=5.0, llm=9.0, gold=9.0),
        AnchorSample(rule=5.0, ml=5.0, llm=9.0, gold=9.0),
        AnchorSample(rule=9.0, ml=9.0, llm=2.0, gold=2.0),
        AnchorSample(rule=9.0, ml=9.0, llm=2.0, gold=2.0),
    ]
    w = fit_match_weights(samples, step=0.25)
    assert w["llm"] >= w["rule"]
    wq = fit_quality_weights(samples, step=0.25)
    assert wq["llm"] >= wq["rule"]
