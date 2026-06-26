"""Layer 2: Embedding backends for resume-JD semantic matching.

Three interchangeable backends, selected by ``MATCH_BACKEND`` (config):

- ``bge-m3``        — FlagEmbedding BGE-M3, 100+ langs, 8192 tokens (design default).
                      Needs ``uv sync --extra embedding``.
- ``sentence-transformers`` — ST multilingual model (lighter cold-start).
                      Needs the ``embedding`` extra.
- ``tfidf``         — scikit-learn TF-IDF + directional JD coverage. Zero heavy
                      deps, always works. Lower accuracy; used as fallback / tests.

All backends expose ``embed(texts) -> List[np.ndarray]`` and ``similarity(a, b)``.
Backends may also implement ``match_score(jd, resume) -> float in [0,1]`` for a
task-specific pair score (the tfidf backend uses directional coverage, which is
far more meaningful than symmetric cosine for long-resume vs short-JD).
The interface mirrors the dual-tower (bi-encoder) stage of the recommended
Retrieve & Re-Rank pipeline (sbert.net).
"""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from src.utils.config import (
    BGE_M3_MODEL_NAME,
    MATCH_BACKEND,
    ST_MODEL_NAME,
)


class Embedder(Protocol):
    """Embedding backend interface."""

    backend: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]."""
        ...

    def match_score(self, jd: str, resume: str) -> float:
        """Optional task-specific pair score in [0,1]. Default: cosine of embeds."""
        ...


# ---------- TF-IDF backend ----------

class TfidfEmbedder:
    """TF-IDF embedder with a directional JD-coverage pair score.

    Symmetric cosine is a poor match metric for long-resume-vs-short-JD (it's
    depressed by the resume's many unique terms). For the pair score we instead
    compute *directional coverage*: the fraction of the JD's distinct (stemmed)
    terms that appear in the resume — i.e. "how much of the JD does the resume
    speak to". ``embed``/``similarity`` remain available for retrieval use cases.
    """

    backend = "tfidf"

    _TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*|[一-鿿]+")
    _STOP = {
        "the", "and", "or", "of", "to", "a", "an", "in", "on", "for", "with",
        "we", "are", "is", "be", "as", "at", "by", "your", "our", "you", "will",
        "have", "has", "this", "that", "from", "not", "but", "if", "it", "its",
        "experience", "year", "years", "team", "work", "ability", "good",
        "strong", "plus", "etc", "must", "should", "able", "related", "field",
        "requirements", "responsibilities", "qualification", "qualifications",
    }

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # local import
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w][\w.+#-]*\b|[一-鿿]",
            ngram_range=(1, 1),
            sublinear_tf=True,
            min_df=1,
        )
        self._fitted = False

    @classmethod
    def _stem(cls, w: str) -> str:
        """Crude suffix stripper so mentor≈mentored, build≈built, deploy≈deployment."""
        w = w.lower().rstrip(".-")
        for suf in ("ment", "tion", "ing", "ers", "ed", "er", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        from src.features.keyword_coverage import _ZH_SKILL_TERMS  # shared dictionary
        out: set[str] = set()
        for m in cls._TOKEN_RE.findall(text):
            if m[0].isascii():
                s = cls._stem(m)
                if len(s) >= 2 and s not in cls._STOP:
                    out.add(s)
            else:
                # CJK: emit any curated skill terms that occur in the text,
                # plus short whole-segments (len 2-3) as fallback keywords.
                if 2 <= len(m) <= 3:
                    out.add(m)
        for term in _ZH_SKILL_TERMS:
            if term in text:
                out.add(term)
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            mat = self._vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            mat = self._vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def match_score(self, jd: str, resume: str) -> float:
        """Directional coverage: fraction of JD tokens present in the resume.

        Latin tokens use stemmed set membership; CJK tokens use substring match
        (whole-segment tokenization is coarse, so '排序模型' should still match a
        resume containing '主导排序模型重构').
        """
        jd_t = self._tokens(jd)
        if not jd_t:
            return 0.0
        res_t = self._tokens(resume)
        covered = 0
        for tok in jd_t:
            if tok[0].isascii():
                if tok in res_t:
                    covered += 1
            elif tok in resume:
                covered += 1
        return covered / len(jd_t)


# ---------- Sentence-Transformers backend ----------

class STEmbedder:
    """Sentence-Transformers bi-encoder. Multilingual, ~470MB."""

    backend = "sentence-transformers"

    def __init__(self, model_name: str = ST_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------- BGE-M3 backend ----------

class BGEM3Embedder:
    """FlagEmbedding BGE-M3 (dense vectors). 100+ langs, 8192 tokens."""

    backend = "bge-m3"

    def __init__(self, model_name: str = BGE_M3_MODEL_NAME) -> None:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        self._model = BGEM3FlagModel(model_name, use_fp16=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def get_embedder(backend: str | None = None) -> Embedder:
    """Factory: return the configured embedder, with graceful fallback.

    Falls back to tfidf if a heavy backend's deps aren't installed OR its model
    weights can't be loaded (e.g. not yet downloaded, network blocked).
    """
    backend = backend or MATCH_BACKEND
    if backend == "bge-m3":
        try:
            return BGEM3Embedder()
        except Exception as e:  # noqa: BLE001 - ImportError, OSError, network, missing weights
            print(f"[embedder] bge-m3 unavailable ({type(e).__name__}); falling back to tfidf")
    if backend == "sentence-transformers":
        try:
            return STEmbedder()
        except Exception as e:  # noqa: BLE001
            print(f"[embedder] sentence-transformers unavailable ({type(e).__name__}); falling back to tfidf")
    return TfidfEmbedder()
