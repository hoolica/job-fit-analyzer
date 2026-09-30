import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from job_analyzer import cli
from job_analyzer.cli import (
    EXIT_ANALYSIS_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    NONE_LISTED,
    NOT_SPECIFIED,
    format_job_analysis,
    main,
)
from job_analyzer.llm_client import (
    LLMAPIError,
    LLMConfigurationError,
    LLMIncompleteOutputError,
    LLMRefusalError,
    LLMValidationError,
)
from job_analyzer.models import JobAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAKE_API_KEY = "sk-ant-test-FAKE-KEY-0123456789"


def full_payload() -> dict:
    return {
        "job_title": "Backend Python Developer",
        "company": "Acme",
        "contract_type": "permanent",
        "skills": [
            {"name": "Python", "category": "language", "required": True},
            {"name": "Docker", "category": "tool", "required": False},
            {"name": "PostgreSQL", "category": "tool", "required": True},
            {"name": "Teamwork", "category": "soft_skill", "required": False},
        ],
        "responsibilities": ["Concevoir des API REST", "Écrire des tests"],
        "experience": {
            "min_years": 2,
            "max_years": None,
            "level": None,
            "source_text": "2+ ans",
        },
        "location": {"city": "Montréal", "country": "Canada", "work_mode": "hybrid"},
        "salary": {
            "min_amount": 75000,
            "max_amount": 90000,
            "currency": "CAD",
            "period": "yearly",
            "source_text": "75 000 $ – 90 000 $",
        },
        "benefits": ["Assurance collective"],
        "requirements": [
            {
                "kind": "education",
                "description": "Diplôme en informatique ou expérience équivalente",
                "importance": "required",
                "source_text": "Diplôme en informatique ou expérience équivalente",
            },
            {
                "kind": "eligibility",
                "description": "Citoyen canadien ou résident permanent",
                "importance": "required",
                "source_text": "Citoyens canadiens et résidents permanents seulement",
            },
            {
                "kind": "language",
                "description": "Anglais",
                "importance": "preferred",
                "source_text": "L'anglais est un atout",
            },
        ],
    }


def empty_payload() -> dict:
    return {
        "job_title": None,
        "company": None,
        "contract_type": None,
        "skills": [],
        "responsibilities": [],
        "experience": {"min_years": None, "max_years": None, "level": None, "source_text": None},
        "location": {"city": None, "country": None, "work_mode": None},
        "salary": {
            "min_amount": None,
            "max_amount": None,
            "currency": None,
            "period": None,
            "source_text": None,
        },
        "benefits": [],
        "requirements": [],
    }


def make_analysis(base: dict | None = None, **changes) -> JobAnalysis:
    """Build a JobAnalysis, overriding nested paths written as "section__field"."""
    data = base if base is not None else full_payload()
    for path, value in changes.items():
        *parents, last = path.split("__")
        target = data
        for key in parents:
            target = target[key]
        target[last] = value
    return JobAnalysis.model_validate(data)


def field_line(output: str, label: str) -> str:
    """Return the value printed next to a label, e.g. "Salary"."""
    for line in output.splitlines():
        if line.startswith(f"{label} "):
            return line[len(label):].strip()
    raise AssertionError(f"label {label!r} not found")


@pytest.fixture(autouse=True)
def fake_load_dotenv(monkeypatch):
    """Never read the developer's real .env in in-process tests."""
    calls = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: calls.append("load_dotenv"))
    return calls


@pytest.fixture
def offer_file(tmp_path):
    path = tmp_path / "offer.txt"
    path.write_text("Développeur Python – CDI – Montréal", encoding="utf-8")
    return path


# --- main(): full flow ---


def test_success_prints_analysis_and_returns_0(offer_file, capsys):
    received = []

    def fake_analyze(text):
        received.append(text)
        return make_analysis()

    code = main([str(offer_file)], analyze=fake_analyze)

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    assert received == ["Développeur Python – CDI – Montréal"]
    assert "Backend Python Developer" in out
    assert str(offer_file) in out
    assert err == ""


def test_dotenv_is_loaded_before_analysis(offer_file, fake_load_dotenv):
    def fake_analyze(text):
        fake_load_dotenv.append("analyze")
        return make_analysis()

    main([str(offer_file)], analyze=fake_analyze)

    assert fake_load_dotenv == ["load_dotenv", "analyze"]


# --- main(): input file ---


@pytest.mark.parametrize(
    "setup, expected_message",
    [
        (lambda p: None, "file not found"),
        (lambda p: p.mkdir(), "not a file"),
        (lambda p: p.write_bytes(b"caf\xe9 au lait"), "not valid UTF-8"),
        (lambda p: p.write_text("", encoding="utf-8"), "is empty"),
        (lambda p: p.write_text("  \n\t ", encoding="utf-8"), "is empty"),
    ],
    ids=["missing", "directory", "not-utf8", "empty", "whitespace-only"],
)
def test_unusable_input_file_returns_usage_error(
    tmp_path, capsys, fake_load_dotenv, setup, expected_message
):
    path = tmp_path / "offer.txt"
    setup(path)

    def must_not_run(text):
        raise AssertionError("analyze must not be called")

    code = main([str(path)], analyze=must_not_run)

    out, err = capsys.readouterr()
    assert code == EXIT_USAGE_ERROR
    assert err.startswith("Error: ")
    assert expected_message in err
    assert out == ""
    assert fake_load_dotenv == []


def test_utf8_bom_is_stripped(tmp_path):
    path = tmp_path / "offer.txt"
    path.write_bytes("﻿Offre".encode("utf-8"))
    received = []

    main([str(path)], analyze=lambda text: received.append(text) or make_analysis())

    assert received == ["Offre"]


def test_missing_argument_returns_usage_error(capsys):
    code = main([])

    assert code == EXIT_USAGE_ERROR
    assert "usage:" in capsys.readouterr().err


# --- main(): analysis errors ---


@pytest.mark.parametrize(
    "error, expected_code",
    [
        (LLMConfigurationError("ANTHROPIC_API_KEY is not set."), EXIT_CONFIG_ERROR),
        (LLMAPIError("Rate limit reached. Try again later."), EXIT_ANALYSIS_ERROR),
        (LLMRefusalError("Claude declined to analyze this offer."), EXIT_ANALYSIS_ERROR),
        (LLMIncompleteOutputError("The response was truncated."), EXIT_ANALYSIS_ERROR),
        (LLMValidationError("Invalid analysis returned by the model."), EXIT_ANALYSIS_ERROR),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
def test_analysis_errors_become_messages_and_exit_codes(offer_file, capsys, error, expected_code):
    def failing_analyze(text):
        raise error

    code = main([str(offer_file)], analyze=failing_analyze)

    out, err = capsys.readouterr()
    assert code == expected_code
    assert err == f"Error: {error}\n"
    assert out == ""


def test_keyboard_interrupt_returns_130(offer_file, capsys):
    def interrupted(text):
        raise KeyboardInterrupt

    code = main([str(offer_file)], analyze=interrupted)

    assert code == EXIT_INTERRUPTED
    assert capsys.readouterr().err == "Interrupted.\n"


# --- format_job_analysis ---


def test_format_full_analysis():
    output = format_job_analysis(make_analysis(), source="offer.txt")

    assert output.startswith("Job analysis: offer.txt\n")
    assert field_line(output, "Title") == "Backend Python Developer"
    assert field_line(output, "Company") == "Acme"
    assert field_line(output, "Contract") == "Permanent"
    assert field_line(output, "Location") == "Montréal, Canada (hybrid)"
    assert field_line(output, "Experience") == "2+ years"
    assert field_line(output, "Salary") == "75,000 – 90,000 CAD per year"
    assert "  - Concevoir des API REST" in output
    assert "Benefits (1)\n  - Assurance collective" in output


def test_format_empty_analysis():
    output = format_job_analysis(make_analysis(empty_payload()))

    for label in ("Title", "Company", "Contract", "Location", "Experience", "Salary"):
        assert field_line(output, label) == NOT_SPECIFIED
    for section in (
        "Required skills",
        "Nice-to-have skills",
        "Other requirements",
        "Responsibilities",
        "Benefits",
    ):
        assert f"{section}\n  {NONE_LISTED}" in output


def test_format_splits_required_and_optional_skills_in_order():
    output = format_job_analysis(make_analysis())

    assert "Required skills (2)\n  - Python (programming language)\n  - PostgreSQL (tool)" in output
    assert "Nice-to-have skills (2)\n  - Docker (tool)\n  - Teamwork (soft skill)" in output


def test_format_requirements_with_importance_kind_and_hard_gate():
    output = format_job_analysis(make_analysis())

    assert (
        "Other requirements (3)\n"
        "  - [required] Education: Diplôme en informatique ou expérience équivalente\n"
        "  - [required] Eligibility: Citoyen canadien ou résident permanent (hard requirement)\n"
        "  - [preferred] Language: Anglais"
    ) in output


def test_preferred_eligibility_is_not_marked_as_hard_requirement():
    analysis = make_analysis(
        requirements=[
            {
                "kind": "eligibility",
                "description": "Security clearance",
                "importance": "preferred",
                "source_text": "Security clearance is an asset",
            }
        ]
    )

    output = format_job_analysis(analysis)

    assert "  - [preferred] Eligibility: Security clearance\n" in output + "\n"
    assert "hard requirement" not in output


@pytest.mark.parametrize(
    "min_years, max_years, level, expected",
    [
        (2, 5, None, "2–5 years"),
        (2, None, None, "2+ years"),
        (None, 1, None, "Up to 1 year"),
        (3, 3, "senior", "3 years · senior"),
        (None, None, "junior", "Junior"),
        (None, None, None, NOT_SPECIFIED),
    ],
)
def test_format_experience(min_years, max_years, level, expected):
    analysis = make_analysis(
        experience__min_years=min_years,
        experience__max_years=max_years,
        experience__level=level,
    )

    assert field_line(format_job_analysis(analysis), "Experience") == expected


@pytest.mark.parametrize(
    "min_amount, max_amount, currency, period, expected",
    [
        (75000, 90000, "CAD", "yearly", "75,000 – 90,000 CAD per year"),
        (500, None, "EUR", "daily", "From 500 EUR per day"),
        (None, 30, None, "hourly", "Up to 30 per hour"),
        (60000, 60000, "USD", None, "60,000 USD"),
        (None, None, "EUR", "yearly", NOT_SPECIFIED),
    ],
)
def test_format_salary(min_amount, max_amount, currency, period, expected):
    analysis = make_analysis(
        salary__min_amount=min_amount,
        salary__max_amount=max_amount,
        salary__currency=currency,
        salary__period=period,
    )

    assert field_line(format_job_analysis(analysis), "Salary") == expected


@pytest.mark.parametrize(
    "city, country, work_mode, expected",
    [
        ("Paris", None, "on_site", "Paris (on-site)"),
        (None, "Canada", None, "Canada"),
        (None, None, "remote", "Remote"),
    ],
)
def test_format_location(city, country, work_mode, expected):
    analysis = make_analysis(
        location__city=city, location__country=country, location__work_mode=work_mode
    )

    assert field_line(format_job_analysis(analysis), "Location") == expected


# --- Real entry point (subprocess, no Anthropic API call) ---


@pytest.fixture
def project_copy(tmp_path):
    """Copy of main.py and the package, so a temporary .env sits at its root."""
    root = tmp_path / "project"
    shutil.copytree(
        PROJECT_ROOT / "job_analyzer",
        root / "job_analyzer",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(PROJECT_ROOT / "main.py", root / "main.py")
    (root / "offer.txt").write_text("Développeur Python – CDI", encoding="utf-8")
    return root


@pytest.fixture
def fake_api():
    """Local HTTP server standing in for the API: records keys, answers 401."""
    received_keys = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received_keys.append(self.headers.get("x-api-key"))
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps(
                {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
            ).encode()
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received_keys
    server.shutdown()
    server.server_close()


def run_entry_point(
    project: Path, *args: str, base_url: str | None = None, extra_env: dict | None = None
):
    """Run the real main.py with no Anthropic credentials or proxies inherited."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ANTHROPIC_") and not key.lower().endswith("_proxy")
    }
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_entry_point_help(project_copy):
    result = run_entry_point(project_copy, "--help")

    assert result.returncode == EXIT_OK
    assert "JOB_OFFER_FILE" in result.stdout


def test_entry_point_loads_api_key_from_dotenv(project_copy, fake_api):
    base_url, received_keys = fake_api
    (project_copy / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_API_KEY}\n", encoding="utf-8")

    result = run_entry_point(project_copy, "offer.txt", base_url=base_url)

    # The key from .env reached the HTTP request; the fake 401 maps to a config error.
    key_was_sent = received_keys == [FAKE_API_KEY]
    assert key_was_sent, "the API key from .env was not sent to the API"
    assert result.returncode == EXIT_CONFIG_ERROR
    assert "ANTHROPIC_API_KEY was rejected" in result.stderr
    key_was_printed = FAKE_API_KEY in result.stdout + result.stderr
    assert not key_was_printed, "the API key appeared in the CLI output"


def test_empty_shell_variable_does_not_hide_dotenv_key(project_copy, fake_api):
    base_url, received_keys = fake_api
    (project_copy / ".env").write_text(f"ANTHROPIC_API_KEY={FAKE_API_KEY}\n", encoding="utf-8")

    result = run_entry_point(
        project_copy, "offer.txt", base_url=base_url, extra_env={"ANTHROPIC_API_KEY": ""}
    )

    key_was_sent = received_keys == [FAKE_API_KEY]
    assert key_was_sent, "the API key from .env was not used"
    assert result.returncode == EXIT_CONFIG_ERROR
    assert "ANTHROPIC_API_KEY was rejected" in result.stderr


def test_entry_point_without_dotenv_reports_missing_key(project_copy, fake_api):
    base_url, received_keys = fake_api

    result = run_entry_point(project_copy, "offer.txt", base_url=base_url)

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "ANTHROPIC_API_KEY is not set" in result.stderr
    assert received_keys == []
