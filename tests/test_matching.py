"""Tests for Layer 2: semantic matching (tfidf backend, no heavy deps)."""
from src.matching.embedder import TfidfEmbedder, get_embedder
from src.matching.matcher import MatchScorer


def test_embedder_factory_falls_back_to_tfidf():
    emb = get_embedder("tfidf")
    assert emb.backend == "tfidf"


def test_tfidf_similarity_identical_text():
    emb = TfidfEmbedder()
    v = emb.embed(["python pytorch machine learning", "python pytorch machine learning"])
    sim = emb.similarity(v[0], v[1])
    assert sim > 0.99


def test_tfidf_similarity_unrelated_lower():
    emb = TfidfEmbedder()
    v = emb.embed(["python pytorch kubernetes", "gardening cooking painting"])
    sim = emb.similarity(v[0], v[1])
    assert sim < 0.2


def test_match_score_range_and_ordering():
    scorer = MatchScorer(embedder=TfidfEmbedder())
    jd = "Senior ML engineer. Python, PyTorch, Kubernetes, recommendation systems."
    good = "ML engineer with Python, PyTorch, Kubernetes, built recommendation systems."
    weak = "Gardener with cooking and painting skills."
    g_score, _, _ = scorer.score(good, jd)
    w_score, _, _ = scorer.score(weak, jd)
    assert 1.0 <= g_score <= 10.0
    assert g_score > w_score


def test_match_score_disabled_reranker():
    """Reranker unavailable (no FlagEmbedding) must not crash; cosine used."""
    scorer = MatchScorer(embedder=TfidfEmbedder())
    score, sim, backend = scorer.score("python engineer", "python engineer job")
    assert score > 5.0
    assert "tfidf" in backend
