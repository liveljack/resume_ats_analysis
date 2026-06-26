"""Layer 0: Resume / JD parsing & structuring.

Handles PDF and plain-text input, language detection (zh/en), and rule-based
section segmentation that works for both Chinese and English resumes.

The Chinese NER dataset in ``data/chinese_resume_ner`` (BMES) is available for
training a dedicated NER model; here we use lightweight rules so the system
runs with no heavy dependencies. ``ChineseNERLoader`` exposes the dataset for
future model training.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.utils.config import CHINESE_CHAR_RATIO_THRESHOLD
from src.utils.schema import JobDescription, ResumeSections

# CJK Unified Ideographs range.
_CJK_RE = re.compile(r"[一-鿿]")


def detect_language(text: str) -> str:
    """Return 'zh', 'en', or 'mixed' based on CJK character ratio."""
    if not text:
        return "en"
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "en"
    cjk = sum(1 for c in chars if _CJK_RE.match(c))
    ratio = cjk / len(chars)
    if ratio < CHINESE_CHAR_RATIO_THRESHOLD:
        return "en"
    # A Chinese resume typically mixes in English tech terms; treat >=0.3 as zh.
    return "zh" if ratio >= 0.3 else "mixed"


def extract_text(path: str | Path) -> str:
    """Extract text from a .txt or .pdf file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "pdfplumber is required for PDF parsing. Install with: uv sync"
        ) from exc
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


# Section heading patterns. Order matters: first match wins per heading.
# Each entry: (canonical_name, [regex patterns]).
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("summary", [r"^summary\s*$", r"^profile\s*$", r"^objective\s*$",
                 r"^个人简介", r"^自我评价", r"^求职意向", r"^个人总结"]),
    ("experience", [r".*experience.*", r".*employment.*", r".*work history.*",
                    r"^工作经历", r"^工作经验", r"^实习经历"]),
    ("education", [r"^education.*", r"^academic.*",
                   r"^教育背景", r"^教育经历", r"^学历"]),
    ("skills", [r"^skills.*", r"^technical skills.*", r"^competenc.*",
                r"^技能", r"^专业技能", r"^核心技能"]),
    ("projects", [r"^projects.*", r"^项目经历", r"^项目经验"]),
    ("contact", [r"^contact.*", r"^联系方式"]),
]


def _match_heading(line: str) -> str | None:
    stripped = line.strip().rstrip(":：").strip()
    if not stripped or len(stripped) > 40:
        return None
    low = stripped.lower()
    for canonical, patterns in _SECTION_PATTERNS:
        for p in patterns:
            if re.search(p, low):
                return canonical
    return None


def _split_contact_and_name(lines: list[str]) -> tuple[str, str, str]:
    """Heuristic: first non-empty short line is the name; lines with @/phone are contact."""
    name = ""
    contact_parts: list[str] = []
    body_start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if not name and len(s) <= 40 and not re.search(r"[@\d]", s):
            name = s
            body_start = i + 1
            continue
        if re.search(r"[@\d\+]", s) and "@" in s or re.search(r"\+?\d[\d\- ]{6,}", s):
            contact_parts.append(s)
            body_start = i + 1
            continue
        break
    return name, " | ".join(contact_parts), "\n".join(lines[body_start:])


def parse_resume(text: str, resume_id: str = "") -> ResumeSections:
    """Parse raw resume text into structured sections."""
    lang = detect_language(text)
    lines = text.splitlines()

    name, contact, body_text = _split_contact_and_name(lines)
    if not body_text.strip():
        body_text = text

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for ln in body_text.splitlines():
        heading = _match_heading(ln)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(ln)

    sections_found = list(sections.keys())
    get = lambda k: "\n".join(sections.get(k, [])).strip()

    # Contact may live only in the header; also scan body for an email line.
    if not contact:
        for ln in lines:
            if "@" in ln:
                contact = ln.strip()
                break

    return ResumeSections(
        name=name,
        contact=contact,
        summary=get("summary"),
        experience=get("experience"),
        education=get("education"),
        skills=get("skills"),
        raw_text=text,
        language=lang,
        sections_found=sections_found,
    )


def parse_jd(text: str) -> JobDescription:
    """Parse a job description into requirements / responsibilities."""
    lang = detect_language(text)
    req_patterns = [r"requirements?", r"qualifications?", r"岗位要求", r"任职要求", r"职位要求"]
    resp_patterns = [r"responsibilities", r"what you.?ll do", r"岗位职责", r"工作职责", r"职责"]

    def _grab(patterns: list[str]) -> str:
        for i, ln in enumerate(text.splitlines()):
            low = ln.strip().lower().rstrip(":：")
            if any(re.search(p, low) for p in patterns):
                buf: list[str] = []
                for nxt in text.splitlines()[i + 1:]:
                    s = nxt.strip()
                    if not s:
                        if buf:
                            break
                        continue
                    # stop at next heading-like line
                    if _match_heading(nxt) or any(
                        re.search(p, s.lower().rstrip(":：")) for p in req_patterns + resp_patterns
                    ):
                        break
                    buf.append(s)
                if buf:
                    return "\n".join(buf)
        return ""

    return JobDescription(
        raw_text=text,
        requirements=_grab(req_patterns),
        responsibilities=_grab(resp_patterns),
        language=lang,
    )


class ChineseNERLoader:
    """Loader for the Chinese resume NER dataset (BMES format).

    Reads ``data/chinese_resume_ner/*.char.bmes``. Each line is ``char TAG``;
    blank line separates sentences. Used to train/evaluate a Chinese NER tagger.
    """

    ENTITY_TYPES = ["NAME", "ORG", "TITLE", "EDU", "CONT", "PRO", "LOC", "RACE"]

    def __init__(self, data_dir: str | Path | None = None) -> None:
        from src.utils.config import CHINESE_NER_DIR
        self.data_dir = Path(data_dir) if data_dir else CHINESE_NER_DIR

    def load(self, split: str = "train") -> list[list[tuple[str, str]]]:
        path = self.data_dir / f"{split}.char.bmes"
        if not path.exists():
            raise FileNotFoundError(f"NER split not found: {path}")
        sentences: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                cur.append((parts[0], parts[1]))
        if cur:
            sentences.append(cur)
        return sentences

    def iter_entities(self, split: str = "train") -> Iterable[tuple[str, str]]:
        """Yield (entity_text, entity_type) by decoding BMES spans."""
        for sent in self.load(split):
            text = ""
            etype: str | None = None
            for ch, tag in sent:
                if tag == "O":
                    if etype and text:
                        yield text, etype
                    text, etype = "", None
                    continue
                prefix, _, typ = tag.partition("-")
                if prefix == "B":
                    if etype and text:
                        yield text, etype
                    text, etype = ch, typ
                elif prefix in ("M", "E"):
                    text += ch
            if etype and text:
                yield text, etype
