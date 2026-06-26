"""Tests for Layer 3: LLM judge (JSON parsing + no-API-key graceful skip).

These do not call the LLM API; they cover the rubric-output parsing logic and
the graceful degradation when no API token is configured.
"""
import os

from src.judging.llm_judge import _extract_json, _parse_judgment, judge_quality


def test_extract_json_direct():
    raw = '{"overall": 8.5, "dimensions": [{"name": "quantification", "score": 9, "reason": "x"}], "reasons": ["a"]}'
    d = _extract_json(raw)
    assert d is not None
    assert d["overall"] == 8.5


def test_extract_json_with_surrounding_text():
    raw = 'Here is my verdict:\n```json\n{"overall": 7.0, "dimensions": [], "reasons": []}\n```\nThanks.'
    d = _extract_json(raw)
    assert d is not None
    assert d["overall"] == 7.0


def test_extract_json_invalid_returns_none():
    assert _extract_json("not json at all") is None
    assert _extract_json("{broken") is None


def test_parse_judgment_clamps_scores():
    raw = {"overall": 15.0, "dimensions": [{"name": "x", "score": -3, "reason": ""}]}
    score, dims, reasons = _parse_judgment(raw)
    assert score == 10.0  # clamped to max
    assert dims[0].score == 0.0  # clamped to min


def test_parse_judgment_overall_floor():
    raw = {"overall": -5.0, "dimensions": []}
    score, _, _ = _parse_judgment(raw)
    assert score == 1.0  # clamped to min 1


def test_judge_returns_none_without_api_key(monkeypatch):
    # Ensure no auth token is present.
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = judge_quality("Some resume text")
    assert out is None  # graceful skip; fusion falls back to rule + ML
