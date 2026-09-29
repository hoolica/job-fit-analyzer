import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from job_analyzer import llm_client
from job_analyzer.llm_client import (
    DEFAULT_MODEL,
    OUTPUT_FORMAT,
    SYSTEM_PROMPT,
    LLMAPIError,
    LLMClientError,
    LLMConfigurationError,
    LLMIncompleteOutputError,
    LLMRefusalError,
    LLMValidationError,
    create_client,
    extract_job_analysis,
)
from job_analyzer.models import JobAnalysis

JOB_TEXT = "Développeur Python H/F - CDI - Paris - 45-55k€"

VALID_ANALYSIS = {
    "job_title": "Développeur Python",
    "company": None,
    "contract_type": "permanent",
    "skills": [{"name": "Python", "category": "language", "required": True}],
    "responsibilities": [],
    "experience": {
        "min_years": None,
        "max_years": None,
        "level": None,
        "source_text": None,
    },
    "location": {"city": "Paris", "country": None, "work_mode": None},
    "salary": {
        "min_amount": 45000,
        "max_amount": 55000,
        "currency": "EUR",
        "period": "yearly",
        "source_text": "45-55k€",
    },
    "benefits": [],
}


class FakeMessages:
    """Stands in for client.messages: records calls, returns or raises."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response, error)


def make_response(text=None, stop_reason="end_turn"):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def valid_response():
    return make_response(json.dumps(VALID_ANALYSIS))


def http_response(status_code: int) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status_code, request=request)


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    """Fail loudly if a test ever tries to build a real Anthropic client."""

    def forbidden(*args, **kwargs):
        raise AssertionError("A real Anthropic client must not be created in tests")

    monkeypatch.setattr(llm_client.anthropic, "Anthropic", forbidden)


# --- Successful extraction ---


def test_returns_validated_job_analysis():
    client = FakeClient(valid_response())

    analysis = extract_job_analysis(JOB_TEXT, client=client)

    assert isinstance(analysis, JobAnalysis)
    assert analysis.contract_type == "permanent"
    assert analysis.salary.min_amount == 45000


def test_request_parameters():
    client = FakeClient(valid_response())

    extract_job_analysis(JOB_TEXT, client=client)

    [call] = client.messages.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"] == [
        {"role": "user", "content": f"<job_offer>\n{JOB_TEXT}\n</job_offer>"}
    ]
    assert call["output_config"]["format"] == OUTPUT_FORMAT


def test_default_model_is_sonnet():
    assert DEFAULT_MODEL == "claude-sonnet-5-5"


def test_model_is_configurable():
    client = FakeClient(valid_response())

    extract_job_analysis(JOB_TEXT, client=client, model="claude-opus-5-5")

    assert client.messages.calls[0]["model"] == "claude-opus-5-5"


def test_output_schema_matches_job_analysis():
    schema = OUTPUT_FORMAT["schema"]

    assert OUTPUT_FORMAT["type"] == "json_schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(JobAnalysis.model_fields)


# --- Input validation ---


@pytest.mark.parametrize("job_text", ["", "   \n\t"])
def test_empty_job_text_is_rejected(job_text):
    client = FakeClient(valid_response())

    with pytest.raises(ValueError):
        extract_job_analysis(job_text, client=client)

    assert client.messages.calls == []


# --- stop_reason handling ---


def test_refusal_raises_refusal_error():
    client = FakeClient(make_response(None, stop_reason="refusal"))

    with pytest.raises(LLMRefusalError):
        extract_job_analysis(JOB_TEXT, client=client)


def test_max_tokens_raises_incomplete_error_before_validation():
    truncated = json.dumps(VALID_ANALYSIS)[:50]
    client = FakeClient(make_response(truncated, stop_reason="max_tokens"))

    with pytest.raises(LLMIncompleteOutputError):
        extract_job_analysis(JOB_TEXT, client=client)


# --- Output validation ---


def test_missing_text_block_raises_validation_error():
    client = FakeClient(make_response(None))

    with pytest.raises(LLMValidationError, match="no JSON output"):
        extract_job_analysis(JOB_TEXT, client=client)


def test_malformed_json_raises_validation_error():
    client = FakeClient(make_response("{not json"))

    with pytest.raises(LLMValidationError):
        extract_job_analysis(JOB_TEXT, client=client)


def test_business_rule_violation_raises_validation_error():
    payload = json.loads(json.dumps(VALID_ANALYSIS))
    payload["salary"]["min_amount"] = 60000
    client = FakeClient(make_response(json.dumps(payload)))

    with pytest.raises(LLMValidationError, match="min_amount must be less than or equal"):
        extract_job_analysis(JOB_TEXT, client=client)


def test_negative_value_raises_validation_error():
    payload = json.loads(json.dumps(VALID_ANALYSIS))
    payload["experience"]["min_years"] = -1
    client = FakeClient(make_response(json.dumps(payload)))

    with pytest.raises(LLMValidationError):
        extract_job_analysis(JOB_TEXT, client=client)


# --- API errors ---


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            anthropic.AuthenticationError(
                "invalid x-api-key", response=http_response(401), body=None
            ),
            LLMConfigurationError,
        ),
        (
            anthropic.RateLimitError(
                "rate limited", response=http_response(429), body=None
            ),
            LLMAPIError,
        ),
        (
            anthropic.BadRequestError(
                "bad request", response=http_response(400), body=None
            ),
            LLMAPIError,
        ),
        (
            anthropic.InternalServerError(
                "overloaded", response=http_response(529), body=None
            ),
            LLMAPIError,
        ),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            LLMAPIError,
        ),
    ],
)
def test_sdk_errors_are_wrapped(error, expected):
    client = FakeClient(error=error)

    with pytest.raises(expected) as exc_info:
        extract_job_analysis(JOB_TEXT, client=client)

    assert exc_info.value.__cause__ is error


def test_all_module_errors_share_a_base_class():
    for error_class in (
        LLMConfigurationError,
        LLMAPIError,
        LLMRefusalError,
        LLMIncompleteOutputError,
        LLMValidationError,
    ):
        assert issubclass(error_class, LLMClientError)


# --- Client creation ---


def test_create_client_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        create_client()


def test_create_client_with_empty_api_key_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(LLMConfigurationError):
        create_client()


def test_create_client_with_api_key_builds_client(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", lambda: sentinel)

    assert create_client() is sentinel


def test_missing_api_key_fails_before_any_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError):
        extract_job_analysis(JOB_TEXT)
