"""Layer 2: Resume-JD semantic matcher (dual-tower + optional cross-encoder rerank).

- ``MatchScorer.score(resume_text, jd_text)`` -> 0-10 score + raw similarity.
- Dual-tower: embed both, cosine similarity.
- Rerank (optional): if a cross-encoder is available, refine the pair score.

Similarity in [0,1] is mapped to 1-10 via a linear map (clamped), since a resume
and JD that are semantically near-identical should score ~10 and unrelated ~1.
"""
from __future__ import annotations

import numpy as np

from src.matching.embedder import Embedder, get_embedder
from src.utils.config import RERANKER_MODEL_NAME


class Reranker:
    """Cross-encoder reranker (bge-reranker-v2-m3). Optional; lazily loaded.

    Uses sentence_transformers.CrossEncoder (robust across transformers versions;
    FlagReranker breaks on transformers>=5 due to tokenizer API changes).
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passage: str) -> float:
        # CrossEncoder.predict returns a sigmoid probability in [0,1] for this model.
        raw = self._model.predict([(query, passage)])
        return float(raw[0])


class MatchScorer:
    """Compute the ML semantic match score between a resume and a JD."""

    def __init__(self, embedder: Embedder | None = None, reranker: Reranker | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.reranker = reranker
        self._reranker_attempted = False

    def _maybe_load_reranker(self) -> None:
        if self._reranker_attempted:
            return
        self._reranker_attempted = True
        try:
            self.reranker = Reranker()
        except Exception:
            self.reranker = None

    def similarity(self, resume_text: str, jd_text: str) -> float:
        emb = self.embedder.embed([resume_text, jd_text])
        return self.embedder.similarity(emb[0], emb[1])

    def score(self, resume_text: str, jd_text: str) -> tuple[float, float, str]:
        """Return (0-10 score, raw similarity, backend info)."""
        # Prefer a task-specific pair score if the backend provides one
        # (tfidf uses directional coverage; embedding backends fall back to cosine).
        backend = self.embedder.backend
        if hasattr(self.embedder, "match_score"):
            try:
                sim = float(self.embedder.match_score(jd_text, resume_text))
            except Exception:
                sim = self.similarity(resume_text, jd_text)
        else:
            sim = self.similarity(resume_text, jd_text)

        # Optional rerank stage: cross-encoder probability overrides the pair score.
        if sim >= 0.0:  # only rerank plausible candidates (save cost)
            self._maybe_load_reranker()
            if self.reranker is not None:
                try:
                    prob = self.reranker.score(jd_text, resume_text)
                    sim = prob  # [0,1]
                    backend += "+rerank"
                except Exception:
                    pass

        # Map [0,1] -> [1,10]. Clamp to [0,1] first.
        s = max(0.0, min(1.0, sim))
        score10 = 1.0 + s * 9.0
        return round(score10, 2), round(sim, 4), backend
