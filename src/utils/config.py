"""Central configuration: paths, model names, fusion weights, rubric weights."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JDS_DIR = DATA_DIR / "sample_jds"
ANCHOR_SET_DIR = DATA_DIR / "anchor_set"
CHINESE_NER_DIR = DATA_DIR / "chinese_resume_ner"

# --- Embedding / matching model ---
# Backend priority: "bge-m3" (FlagEmbedding) > "sentence-transformers" > "tfidf" (fallback).
# Set via env var MATCH_BACKEND. Defaults to "tfidf" so the system runs with zero heavy deps;
# install the `embedding` extra and set MATCH_BACKEND=sentence-transformers or bge-m3 for accuracy.
MATCH_BACKEND = os.environ.get("MATCH_BACKEND", "tfidf")

# Sentence-Transformers multilingual model (zh+en, ~470MB). Lighter than bge-m3, good cold-start.
ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
# BGE-M3: the design's recommended high-accuracy multilingual embedding (100+ langs, 8192 tokens).
BGE_M3_MODEL_NAME = os.environ.get("BGE_M3_MODEL_NAME", "BAAI/bge-m3")
# Cross-encoder reranker (paired with bge-m3).
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# --- LLM judge ---
# Reuses the project's Anthropic-compatible client (GLM via Zhipu / BigModel).
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.6")
# Number of self-consistency samples for the LLM judge (majority vote / mean).
LLM_SELF_CONSISTENCY_K = int(os.environ.get("LLM_SELF_CONSISTENCY_K", "3"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1600"))

# --- Fusion weights (initial values; calibrate against anchor set) ---
# quality = w_rule * rule_quality + w_llm * llm_quality  (ML not used for quality)
FUSION_QUALITY = {
    "rule": float(os.environ.get("FQ_RULE", "0.3")),
    "llm": float(os.environ.get("FQ_LLM", "0.7")),
}
# match = w_rule * rule_keyword + w_ml * ml_match + w_llm * llm_match
FUSION_MATCH = {
    "rule": float(os.environ.get("FM_RULE", "0.2")),
    "ml": float(os.environ.get("FM_ML", "0.3")),
    "llm": float(os.environ.get("FM_LLM", "0.5")),
}

# Pass/fail thresholds (binary "达标" judgment for evaluation).
QUALITY_PASS_THRESHOLD = 6.0
MATCH_PASS_THRESHOLD = 6.0

# Language detection: ratio of CJK characters above which text is treated as Chinese.
CHINESE_CHAR_RATIO_THRESHOLD = 0.15


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a weight dict so values sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}
