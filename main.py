"""CLI entry point for the resume analysis system.

Examples:
  # Score a single resume's quality (1-10)
  python main.py quality --resume data/sample_resumes/good_en.txt

  # Score resume-JD match (1-10)
  python main.py match --resume data/sample_resumes/good_en.txt --jd data/sample_jds/ml_engineer_en.txt

  # Full analysis (quality + match + pass/fail), JSON output
  python main.py analyze --resume data/sample_resumes/good_zh.txt --jd data/sample_jds/ml_engineer_zh.txt

  # Run the test suite over sample resumes (sanity / demo)
  python main.py demo

Environment:
  MATCH_BACKEND         tfidf | sentence-transformers | bge-m3  (default: tfidf)
  ANTHROPIC_AUTH_TOKEN  enables the LLM judge (GLM via Zhipu/BigModel)
  ANTHROPIC_BASE_URL    base URL for the Anthropic-compatible endpoint
"""
from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import analyze_file
from src.utils.config import (
    SAMPLE_JDS_DIR,
    SAMPLE_RESUMES_DIR,
)


def _print(result) -> None:
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


def cmd_quality(args) -> int:
    res = analyze_file(args.resume, use_ml=False, use_llm=args.llm)
    _print(res)
    return 0


def cmd_match(args) -> int:
    if not args.jd:
        print("--jd is required for match", file=sys.stderr)
        return 2
    res = analyze_file(args.resume, args.jd, use_ml=not args.no_ml, use_llm=args.llm)
    _print(res)
    return 0


def cmd_analyze(args) -> int:
    res = analyze_file(args.resume, args.jd, use_ml=not args.no_ml, use_llm=args.llm)
    _print(res)
    return 0


def cmd_demo(args) -> int:
    """Run analysis over all sample resumes and print a compact table."""
    pairs = [
        ("good_en", "ml_engineer_en"),
        ("weak_en", "ml_engineer_en"),
        ("good_zh", "ml_engineer_zh"),
        ("weak_zh", "ml_engineer_zh"),
    ]
    print(f"{'resume':10} {'quality':>8} {'match':>8} {'pass':>6}")
    print("-" * 40)
    for r, j in pairs:
        rp = SAMPLE_RESUMES_DIR / f"{r}.txt"
        jp = SAMPLE_JDS_DIR / f"{j}.txt"
        res = analyze_file(rp, jp, use_ml=True, use_llm=args.llm)
        q = res.quality.score
        m = res.match.score if res.match else 0.0
        p = "yes" if res.passed else "no"
        print(f"{r:10} {q:>8.2f} {m:>8.2f} {p:>6}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="resume-analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pq = sub.add_parser("quality", help="Score resume quality 1-10")
    pq.add_argument("--resume", required=True, help="Path to resume (.txt or .pdf)")
    pq.add_argument("--llm", action="store_true", help="Enable LLM judge (needs API key)")
    pq.set_defaults(func=cmd_quality)

    pm = sub.add_parser("match", help="Score resume-JD match 1-10")
    pm.add_argument("--resume", required=True)
    pm.add_argument("--jd", help="Path to job description")
    pm.add_argument("--no-ml", action="store_true", help="Disable ML semantic matching")
    pm.add_argument("--llm", action="store_true", help="Enable LLM judge")
    pm.set_defaults(func=cmd_match)

    pa = sub.add_parser("analyze", help="Full analysis (quality + match + pass/fail)")
    pa.add_argument("--resume", required=True)
    pa.add_argument("--jd", help="Path to job description")
    pa.add_argument("--no-ml", action="store_true")
    pa.add_argument("--llm", action="store_true", help="Enable LLM judge")
    pa.set_defaults(func=cmd_analyze)

    pd = sub.add_parser("demo", help="Run over sample resumes")
    pd.add_argument("--llm", action="store_true", help="Enable LLM judge")
    pd.set_defaults(func=cmd_demo)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
