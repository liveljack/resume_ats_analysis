"""Layer 1 (rules): JD keyword extraction + coverage on resume.

Deterministic keyword-overlap matching (à la Resume-Matcher's rule layer).
Produces a 0-10 match sub-score and the list of matched / missing keywords.
This is the cheap first-line match signal; the ML embedding layer (src/matching)
adds semantic similarity on top.
"""
from __future__ import annotations

import re
from collections import Counter

from src.utils.schema import JobDescription, ResumeSections

# Stopwords (English + common Chinese filler).
_STOP_EN = {
    "the", "and", "or", "of", "to", "a", "an", "in", "on", "for", "with", "we",
    "are", "is", "be", "as", "at", "by", "your", "our", "you", "will", "have",
    "has", "this", "that", "from", "or", "not", "but", "if", "so", "it", "its",
    "experience", "year", "years", "team", "work", "working", "ability", "good",
    "strong", "nice", "plus", "etc", "must", "should", "able", "related", "field",
    "requirements", "responsibilities", "qualification", "qualifications",
}
_STOP_ZH = {"的", "与", "和", "或", "及", "以及", "在", "为", "是", "有", "能", "可",
            "能够", "具备", "具有", "熟悉", "了解", "掌握", "熟练", "优先", "相关",
            "经验", "能力", "良好", "优秀", "扎实", "职位", "岗位", "招聘", "负责", "职责",
            "年以上", "以上", "以下", "岗位要求", "岗位职责", "任职要求", "职位要求",
            "要求", "职责", "加分项", "nice", "have"}

# Skill / tech tokens worth treating as whole keywords even when multiword.
_SKILL_HINTS = {
    "pytorch", "tensorflow", "python", "java", "kubernetes", "docker", "spark",
    "kafka", "aws", "gcp", "azure", "sql", "nosql", "nlp", "ml", "ai", "xgboost",
    "lightgbm", "airflow", "pycharm", "git", "linux", "scala", "go", "rust",
    "react", "vue", "django", "flask", "fastapi", "hadoop", "hive", "elasticsearch",
}

# Curated Chinese tech/skill terms. Used for CJK keyword extraction: whole-segment
# tokenization is too coarse ("具备推荐系统" as one token), and bigrams produce junk.
# A domain dictionary yields meaningful, comparable keywords + real gap analysis.
_ZH_SKILL_TERMS = [
    "推荐系统", "排序模型", "深度学习", "机器学习", "自然语言处理", "计算机视觉",
    "实时特征", "特征工程", "分布式", "微服务", "高可用", "模型蒸馏", "模型部署",
    "大语言模型", "向量检索", "知识图谱", "数据挖掘", "用户画像", "AB实验",
    "强化学习", "图神经网络", "序列建模", "点击率", "转化率", "留存率",
    "搜索推荐", "广告算法", "风控", "NLP", "CV",
    "PyTorch", "TensorFlow", "Python", "Java", "Go", "Scala", "Rust",
    "Spark", "Flink", "Kafka", "Hadoop", "Hive", "Airflow", "Docker", "Kubernetes",
    "AWS", "GCP", "Azure", "Linux", "Git", "SQL", "Redis", "MySQL",
    "XGBoost", "LightGBM", "Elasticsearch",
]


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed zh/en text.

    Latin: lowercased alnum words. CJK: curated skill-dictionary terms only
    (whole-segment tokenization produces coarse junk like '具备推荐系统'; the
    dictionary yields comparable, meaningful keywords).
    """
    tokens: list[str] = []
    # English / latin words
    for w in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", text):
        w = w.lower().rstrip(".-")
        if len(w) >= 2 and w not in _STOP_EN:
            tokens.append(w)
    # Chinese: dictionary terms found as substrings.
    for term in _ZH_SKILL_TERMS:
        if term in text:
            tokens.append(term)
    return tokens


def extract_jd_keywords(jd: JobDescription, topk: int = 25) -> list[str]:
    """Extract the most salient keywords from a JD.

    Latin tokens via TF ranking (skill-hint boost); Chinese keywords via a curated
    skill dictionary matched as substrings against the JD text.
    """
    text = " ".join([jd.requirements, jd.responsibilities, jd.raw_text])
    tokens = _tokenize(text)
    counts = Counter(tokens)
    scored = sorted(counts.items(), key=lambda kv: (kv[0] in _SKILL_HINTS, kv[1]), reverse=True)

    seen: set[str] = set()
    out: list[str] = []
    for tok, _ in scored:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)

    # Append curated Chinese skill terms actually present in the JD.
    for term in _ZH_SKILL_TERMS:
        if term in text and term not in seen:
            seen.add(term)
            out.append(term)

    return out[:topk]


def keyword_coverage(resume: ResumeSections, jd_keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """Return (0-10 score, matched, missing). Substring match handles CJK + multiword skills."""
    if not jd_keywords:
        return 5.0, [], []
    resume_text = (resume.skills + " " + resume.experience + " " + resume.raw_text).lower()
    resume_tokens = set(_tokenize(resume_text))
    matched: list[str] = []
    missing: list[str] = []
    for kw in jd_keywords:
        if kw in resume_tokens or kw.lower() in resume_text:
            matched.append(kw)
        else:
            missing.append(kw)
    ratio = len(matched) / len(jd_keywords) if jd_keywords else 0.0
    score = round(ratio * 10.0, 1)
    return score, matched, missing
