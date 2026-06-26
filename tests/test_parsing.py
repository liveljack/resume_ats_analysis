"""Tests for Layer 0: parsing & language detection."""
from src.parsing.parser import (
    ChineseNERLoader,
    detect_language,
    parse_jd,
    parse_resume,
)


def test_detect_language():
    assert detect_language("hello world this is english text") == "en"
    assert detect_language("这是一份中文简历，包含多个中文字符") == "zh"
    assert detect_language("") == "en"


def test_parse_resume_english_sections():
    text = (
        "Jane Doe\njane@email.com | +1-555-0100\n\n"
        "SUMMARY\nSenior engineer with 10 years experience.\n\n"
        "EXPERIENCE\n- Led team of 5\n- Built scalable system serving 1M users\n\n"
        "EDUCATION\nB.S. Computer Science, MIT\n\n"
        "SKILLS\nPython, Go, Kubernetes\n"
    )
    r = parse_resume(text)
    assert r.name == "Jane Doe"
    assert "jane@email.com" in r.contact
    assert "senior engineer" in r.summary.lower()
    assert "led team" in r.experience.lower()
    assert "mit" in r.education.lower()
    assert "python" in r.skills.lower()
    assert set(r.sections_found) >= {"summary", "experience", "education", "skills"}
    assert r.language == "en"


def test_parse_resume_chinese():
    text = (
        "王建国\n邮箱：wg@email.com\n\n"
        "工作经历\n- 主导排序模型重构，点击率提升18%\n\n"
        "教育背景\n清华大学\n\n"
        "技能\nPython, PyTorch\n"
    )
    r = parse_resume(text)
    assert r.name == "王建国"
    assert "主导" in r.experience
    assert "清华" in r.education
    assert "pytorch" in r.skills.lower()
    assert r.language == "zh"


def test_parse_jd_splits_requirements():
    text = (
        "Senior ML Engineer\n\n"
        "Requirements:\n- 5+ years Python\n- PyTorch experience\n\n"
        "Responsibilities:\n- Build ranking models\n- Mentor engineers\n"
    )
    jd = parse_jd(text)
    assert "5+ years" in jd.requirements or "python" in jd.requirements.lower()
    assert "ranking" in jd.responsibilities.lower() or "mentor" in jd.responsibilities.lower()


def test_chinese_ner_dataset_loaded():
    """The downloaded BMES dataset is readable and decodes entities."""
    loader = ChineseNERLoader()
    try:
        sents = loader.load("test")
    except FileNotFoundError:
        # Dataset not downloaded in this env — skip gracefully.
        import pytest
        pytest.skip("chinese_resume_ner test split not present")
    assert len(sents) > 0
    entities = list(loader.iter_entities("test"))
    types = {et for _, et in entities}
    # Should contain at least a few of the known entity types.
    assert types & {"NAME", "ORG", "TITLE", "EDU"}
