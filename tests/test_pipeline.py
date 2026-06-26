"""End-to-end pipeline tests (tfidf backend, no LLM key required)."""
from src.pipeline import analyze_text


GOOD_EN = (
    "John Smith\njohn.smith@email.com | +1-415-555-0142\n\n"
    "SUMMARY\nSenior ML Engineer, 7 years experience.\n\n"
    "EXPERIENCE\n- Led ranking redesign, CTR +18%, $4.2M revenue\n"
    "- Built real-time pipeline, 2B events/day, 99.9% uptime\n"
    "- Mentored 4 engineers, cut iteration time 40%\n\n"
    "EDUCATION\nM.S. CS, Stanford\n\n"
    "SKILLS\nPython, PyTorch, TensorFlow, Kubernetes, Spark, Docker, SQL, Kafka\n"
)

WEAK_EN = (
    "Bob\n\nObjective: looking for a job\n\n"
    "EXPERIENCE\n- worked on stuff\n- helped with projects\n\n"
    "SKILLS\nword, excel\n"
)

JD_EN = (
    "Senior ML Engineer\n\nRequirements:\n- 5+ years Python, PyTorch\n"
    "- recommendation systems, ranking models\n- Kubernetes, Spark, AWS\n"
    "- Docker deployment\n\nResponsibilities:\n- Build ranking models\n- Mentor engineers\n"
)


def test_quality_score_good_vs_weak():
    good = analyze_text(GOOD_EN, use_ml=False, use_llm=False)
    weak = analyze_text(WEAK_EN, use_ml=False, use_llm=False)
    assert good.quality.score > weak.quality.score
    assert good.quality.score >= 6.0
    assert weak.quality.score <= 5.0


def test_match_score_good_vs_weak():
    good = analyze_text(GOOD_EN, JD_EN, use_ml=True, use_llm=False)
    weak = analyze_text(WEAK_EN, JD_EN, use_ml=True, use_llm=False)
    assert good.match is not None and weak.match is not None
    assert good.match.score > weak.match.score


def test_passed_flag_set_when_jd_given():
    good = analyze_text(GOOD_EN, JD_EN, use_ml=True, use_llm=False)
    assert good.passed is True
    weak = analyze_text(WEAK_EN, JD_EN, use_ml=True, use_llm=False)
    assert weak.passed is False


def test_passed_none_without_jd():
    res = analyze_text(GOOD_EN, jd_text=None, use_ml=False, use_llm=False)
    assert res.match is None
    assert res.passed is None


def test_scores_in_valid_range():
    res = analyze_text(GOOD_EN, JD_EN, use_ml=True, use_llm=False)
    assert 1.0 <= res.quality.score <= 10.0
    assert 1.0 <= res.match.score <= 10.0


def test_chinese_pipeline():
    zh_resume = (
        "王建国\n邮箱：wg@email.com\n\n"
        "工作经历\n- 主导排序模型重构，点击率提升18%，年化收入4200万\n"
        "- 建设实时特征管线，日处理20亿事件\n\n"
        "教育背景\n清华大学\n\n"
        "技能\nPython, PyTorch, Kubernetes, Spark, Docker\n"
    )
    zh_jd = (
        "高级算法工程师\n岗位要求：\n- 5年以上Python、PyTorch经验\n"
        "- 推荐系统、排序模型\n- Kubernetes、Spark\n岗位职责：\n- 建设排序模型\n"
    )
    res = analyze_text(zh_resume, zh_jd, use_ml=True, use_llm=False)
    assert 1.0 <= res.quality.score <= 10.0
    assert 1.0 <= res.match.score <= 10.0
    assert res.passed is True
