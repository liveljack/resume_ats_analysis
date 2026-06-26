"""Layer 3: LLM-as-judge with rubric scoring + self-consistency.

- Uses the project's Anthropic-compatible client (GLM via Zhipu/BigModel),
  configurable via ``LLM_MODEL`` / env (see src/utils/config.py).
- Self-consistency (arXiv:2203.11171): samples K times, takes the median overall
  score and merges dimension reasons. Reduces variance meaningfully.
- Graceful no-op when no API key is set: returns None so the fusion layer falls
  back to rule + ML signals only.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass

from src.judging.prompts import (
    MATCH_RUBRIC,
    QUALITY_RUBRIC,
    build_match_user_prompt,
    build_quality_user_prompt,
)
from src.utils.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SELF_CONSISTENCY_K,
    LLM_TEMPERATURE,
)
from src.utils.schema import DimensionScore


@dataclass
class JudgeOutput:
    score: float
    dimensions: list[DimensionScore]
    reasons: list[str]
    samples: int


def _get_client():
    """Return an Anthropic client or None if no auth token is configured."""
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        return None
    try:
        import anthropic  # type: ignore
    except ImportError:
        return None
    kwargs = {"api_key": token}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def _extract_json(text: str) -> dict | None:
    """Robustly pull a JSON object out of an LLM response."""
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_judgment(raw: dict) -> tuple[float, list[DimensionScore], list[str]]:
    overall = float(raw.get("overall", 0))
    overall = max(1.0, min(10.0, overall))
    dims: list[DimensionScore] = []
    for d in raw.get("dimensions", []):
        dims.append(
            DimensionScore(
                name=str(d.get("name", "")),
                score=max(0.0, min(10.0, float(d.get("score", 0)))),
                weight=0.0,
                reason=str(d.get("reason", "")),
            )
        )
    reasons = [str(r) for r in raw.get("reasons", [])]
    return overall, dims, reasons


def _call_once(client, system: str, user: str) -> dict | None:
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # collect text content
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    parsed = _extract_json(text)
    if parsed is None:
        return None
    return parsed


def _judge(client, system: str, user: str, k: int) -> JudgeOutput | None:
    if client is None:
        return None
    samples: list[dict] = []
    for _ in range(max(1, k)):
        try:
            raw = _call_once(client, system, user)
        except Exception:
            raw = None
        if raw:
            samples.append(raw)
    if not samples:
        return None

    overalls = []
    all_dims: dict[str, list[float]] = {}
    reasons: list[str] = []
    for raw in samples:
        o, dims, rs = _parse_judgment(raw)
        overalls.append(o)
        for d in dims:
            all_dims.setdefault(d.name, []).append(d.score)
        reasons.extend(rs)

    median_score = round(statistics.median(overalls), 2)
    merged_dims = [
        DimensionScore(
            name=name,
            score=round(statistics.mean(scores), 2),
            weight=0.0,
            reason="",
        )
        for name, scores in all_dims.items()
    ]
    # Dedup reasons preserving order.
    seen = set()
    dedup_reasons = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            dedup_reasons.append(r)

    return JudgeOutput(
        score=median_score,
        dimensions=merged_dims,
        reasons=dedup_reasons[:5],
        samples=len(samples),
    )


def judge_quality(resume_text: str, k: int | None = None) -> JudgeOutput | None:
    client = _get_client()
    return _judge(
        client,
        QUALITY_RUBRIC,
        build_quality_user_prompt(resume_text),
        LLM_SELF_CONSISTENCY_K if k is None else k,
    )


def judge_match(resume_text: str, jd_text: str, k: int | None = None) -> JudgeOutput | None:
    client = _get_client()
    return _judge(
        client,
        MATCH_RUBRIC,
        build_match_user_prompt(resume_text, jd_text),
        LLM_SELF_CONSISTENCY_K if k is None else k,
    )
