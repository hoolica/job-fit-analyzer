"""Command-line interface: read a job offer file and print its analysis.

This module only handles I/O and presentation. Analysis is delegated to
analyzer.py; the LLM client is never called from here (only its exception
classes are imported, to turn them into user-facing messages).
"""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from job_analyzer.analyzer import analyze_job_offer
from job_analyzer.llm_client import LLMClientError, LLMConfigurationError
from job_analyzer.models import (
    Experience,
    JobAnalysis,
    Location,
    Requirement,
    Salary,
    Skill,
)

EXIT_OK = 0
EXIT_ANALYSIS_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_CONFIG_ERROR = 3
EXIT_INTERRUPTED = 130

NOT_SPECIFIED = "Not specified"
NONE_LISTED = "(none listed)"

CONTRACT_LABELS = {
    "permanent": "Permanent",
    "fixed_term": "Fixed-term",
    "internship": "Internship",
    "apprenticeship": "Apprenticeship",
    "freelance": "Freelance",
}
WORK_MODE_LABELS = {"on_site": "on-site", "hybrid": "hybrid", "remote": "remote"}
PERIOD_LABELS = {
    "yearly": "per year",
    "monthly": "per month",
    "daily": "per day",
    "hourly": "per hour",
}
CATEGORY_LABELS = {
    "technical": "technical",
    "tool": "tool",
    "language": "programming language",
    "soft_skill": "soft skill",
}
REQUIREMENT_KIND_LABELS = {
    "education": "Education",
    "eligibility": "Eligibility",
    "language": "Language",
    "other": "Other",
}

Analyze = Callable[[str], JobAnalysis]


class InputFileError(Exception):
    """The job offer file cannot be used."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Analyze a job offer text file with Claude and print a structured summary.",
    )
    parser.add_argument(
        "job_offer_file",
        metavar="JOB_OFFER_FILE",
        type=Path,
        help="path to a UTF-8 text file containing the job offer",
    )
    return parser


def main(argv: list[str] | None = None, *, analyze: Analyze = analyze_job_offer) -> int:
    """Run the CLI and return an exit code. Never calls sys.exit itself."""
    try:
        return _run(argv, analyze)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


def _run(argv: list[str] | None, analyze: Analyze) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        # argparse exits with 0 for --help and 2 for invalid arguments.
        return e.code if isinstance(e.code, int) else EXIT_USAGE_ERROR

    try:
        job_text = _read_job_offer(args.job_offer_file)
    except InputFileError as e:
        _print_error(str(e))
        return EXIT_USAGE_ERROR

    # An empty variable in the shell would stop load_dotenv from using .env.
    if os.environ.get("ANTHROPIC_API_KEY") == "":
        del os.environ["ANTHROPIC_API_KEY"]
    load_dotenv()

    try:
        analysis = analyze(job_text)
    except LLMConfigurationError as e:
        _print_error(str(e))
        return EXIT_CONFIG_ERROR
    except LLMClientError as e:
        _print_error(str(e))
        return EXIT_ANALYSIS_ERROR

    print(format_job_analysis(analysis, source=str(args.job_offer_file)))
    return EXIT_OK


def _read_job_offer(path: Path) -> str:
    """Read a UTF-8 job offer (a leading BOM is ignored); reject unusable files."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise InputFileError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise InputFileError(f"not a file: {path}") from None
    except PermissionError:
        raise InputFileError(f"permission denied: {path}") from None
    except UnicodeDecodeError:
        raise InputFileError(f"{path} is not valid UTF-8 text") from None

    if not text.strip():
        raise InputFileError(f"{path} is empty")
    return text


def _print_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


# --- Formatting ---


def format_job_analysis(analysis: JobAnalysis, *, source: str | None = None) -> str:
    """Render a JobAnalysis as plain text. Pure function."""
    title = f"Job analysis: {source}" if source else "Job analysis"
    lines = [title, "=" * len(title), ""]

    fields = [
        ("Title", analysis.job_title or NOT_SPECIFIED),
        ("Company", analysis.company or NOT_SPECIFIED),
        ("Contract", CONTRACT_LABELS.get(analysis.contract_type, NOT_SPECIFIED)),
        ("Location", _format_location(analysis.location)),
        ("Experience", _format_experience(analysis.experience)),
        ("Salary", _format_salary(analysis.salary)),
    ]
    lines += [f"{label:<12} {value}" for label, value in fields]

    required = [s for s in analysis.skills if s.required]
    optional = [s for s in analysis.skills if not s.required]
    lines += _section("Required skills", [_format_skill(s) for s in required])
    lines += _section("Nice-to-have skills", [_format_skill(s) for s in optional])
    lines += _section(
        "Other requirements", [_format_requirement(r) for r in analysis.requirements]
    )
    lines += _section("Responsibilities", analysis.responsibilities)
    lines += _section("Benefits", analysis.benefits)

    return "\n".join(lines)


def _section(title: str, items: list[str]) -> list[str]:
    if not items:
        return ["", title, f"  {NONE_LISTED}"]
    return ["", f"{title} ({len(items)})"] + [f"  - {item}" for item in items]


def _format_skill(skill: Skill) -> str:
    return f"{skill.name} ({CATEGORY_LABELS[skill.category]})"


def _format_requirement(requirement: Requirement) -> str:
    text = (
        f"[{requirement.importance}] "
        f"{REQUIREMENT_KIND_LABELS[requirement.kind]}: {requirement.description}"
    )
    return f"{text} (hard requirement)" if requirement.is_hard_gate else text


def _format_location(location: Location) -> str:
    place = ", ".join(part for part in (location.city, location.country) if part)
    mode = WORK_MODE_LABELS.get(location.work_mode)
    if place and mode:
        return f"{place} ({mode})"
    if place:
        return place
    if mode:
        return mode.capitalize()
    return NOT_SPECIFIED


def _years(n: int) -> str:
    return f"{n} year" if n == 1 else f"{n} years"


def _format_experience(experience: Experience) -> str:
    low, high = experience.min_years, experience.max_years
    if low is not None and high is not None:
        years = _years(low) if low == high else f"{low}–{high} years"
    elif low is not None:
        years = f"{low}+ years"
    elif high is not None:
        years = f"Up to {_years(high)}"
    else:
        years = None

    level = experience.level
    if years and level:
        return f"{years} · {level}"
    if years:
        return years
    if level:
        return level.capitalize()
    return NOT_SPECIFIED


def _format_salary(salary: Salary) -> str:
    low, high = salary.min_amount, salary.max_amount
    if low is not None and high is not None:
        amount = f"{low:,}" if low == high else f"{low:,} – {high:,}"
    elif low is not None:
        amount = f"From {low:,}"
    elif high is not None:
        amount = f"Up to {high:,}"
    else:
        return NOT_SPECIFIED

    parts = [amount, salary.currency, PERIOD_LABELS.get(salary.period)]
    return " ".join(part for part in parts if part)
