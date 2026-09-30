import pytest
from pydantic import ValidationError

from job_analyzer.models import (
    Experience,
    JobAnalysis,
    Location,
    Requirement,
    Salary,
    Skill,
)


def valid_payload() -> dict:
    return {
        "job_title": "Backend Python Developer",
        "company": "Acme",
        "contract_type": "permanent",
        "skills": [
            {"name": "Python", "category": "language", "required": True},
            {"name": "Docker", "category": "tool", "required": False},
            {"name": "Teamwork", "category": "soft_skill", "required": False},
        ],
        "responsibilities": ["Design REST APIs", "Write tests"],
        "experience": {
            "min_years": 2,
            "max_years": 5,
            "level": "mid",
            "source_text": "2 to 5 years of experience",
        },
        "location": {"city": "Paris", "country": "France", "work_mode": "hybrid"},
        "salary": {
            "min_amount": 45000,
            "max_amount": 55000,
            "currency": "EUR",
            "period": "yearly",
            "source_text": "45-55k EUR gross per year",
        },
        "benefits": ["Meal vouchers", "Extra paid leave"],
        "requirements": [
            {
                "kind": "education",
                "description": "Degree in Computer Science",
                "importance": "required",
                "source_text": "Degree in Computer Science",
            },
        ],
    }


def null_payload() -> dict:
    """Payload where every nullable field is explicitly null."""
    return {
        "job_title": None,
        "company": None,
        "contract_type": None,
        "skills": [],
        "responsibilities": [],
        "experience": {
            "min_years": None,
            "max_years": None,
            "level": None,
            "source_text": None,
        },
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


def set_at(payload: dict, path: tuple, value) -> None:
    """Set a value in a nested payload, following a path of keys/indexes."""
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


# --- Valid cases ---


def test_valid_job_analysis():
    analysis = JobAnalysis.model_validate(valid_payload())

    assert analysis.job_title == "Backend Python Developer"
    assert analysis.skills[0] == Skill(name="Python", category="language", required=True)
    assert analysis.experience.min_years == 2
    assert analysis.location.work_mode == "hybrid"
    assert analysis.salary.max_amount == 55000


def test_all_nullable_fields_accept_explicit_null():
    analysis = JobAnalysis.model_validate(null_payload())

    assert analysis.job_title is None
    assert analysis.experience.level is None
    assert analysis.salary.currency is None


def test_equal_min_and_max_are_accepted():
    payload = valid_payload()
    payload["experience"]["min_years"] = 3
    payload["experience"]["max_years"] = 3
    payload["salary"]["min_amount"] = 50000
    payload["salary"]["max_amount"] = 50000

    JobAnalysis.model_validate(payload)


def test_range_check_skipped_when_one_bound_is_null():
    Experience(min_years=5, max_years=None, level=None, source_text=None)
    Salary(min_amount=None, max_amount=10, currency=None, period=None, source_text=None)


def test_round_trip_through_json():
    analysis = JobAnalysis.model_validate(valid_payload())
    assert JobAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


# --- Range validation ---


def test_min_years_greater_than_max_years_is_rejected():
    payload = valid_payload()
    payload["experience"]["min_years"] = 6
    payload["experience"]["max_years"] = 3

    with pytest.raises(ValidationError, match="min_years must be less than or equal"):
        JobAnalysis.model_validate(payload)


def test_min_amount_greater_than_max_amount_is_rejected():
    payload = valid_payload()
    payload["salary"]["min_amount"] = 60000
    payload["salary"]["max_amount"] = 40000

    with pytest.raises(ValidationError, match="min_amount must be less than or equal"):
        JobAnalysis.model_validate(payload)


# --- Literal validation ---


@pytest.mark.parametrize(
    "path, value",
    [
        (("contract_type",), "cdi"),
        (("skills", 0, "category"), "framework"),
        (("experience", "level"), "expert"),
        (("location", "work_mode"), "full_remote"),
        (("salary", "period"), "weekly"),
        (("requirements", 0, "kind"), "certification"),
        (("requirements", 0, "importance"), "mandatory"),
    ],
)
def test_invalid_literal_is_rejected(path, value):
    payload = valid_payload()
    set_at(payload, path, value)

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == path


# --- Extra fields ---


@pytest.mark.parametrize(
    "path",
    [
        ("unexpected",),
        ("experience", "unexpected"),
        ("location", "unexpected"),
        ("salary", "unexpected"),
        ("skills", 0, "unexpected"),
        ("requirements", 0, "unexpected"),
    ],
)
def test_extra_field_is_forbidden(path):
    payload = valid_payload()
    set_at(payload, path, "value")

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    error = exc_info.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == path


# --- Negative numbers ---


@pytest.mark.parametrize(
    "path",
    [
        ("experience", "min_years"),
        ("experience", "max_years"),
        ("salary", "min_amount"),
        ("salary", "max_amount"),
    ],
)
def test_negative_value_is_rejected(path):
    payload = null_payload()
    set_at(payload, path, -1)

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    error = exc_info.value.errors()[0]
    assert error["type"] == "greater_than_equal"
    assert error["loc"] == path


# --- Required-but-nullable fields ---


@pytest.mark.parametrize(
    "model, field",
    [
        (JobAnalysis, "job_title"),
        (Experience, "min_years"),
        (Location, "city"),
        (Salary, "currency"),
        (JobAnalysis, "requirements"),
    ],
)
def test_nullable_fields_are_required_in_schema(model, field):
    assert field in model.model_json_schema()["required"]


def test_missing_nullable_field_is_rejected():
    payload = valid_payload()
    del payload["location"]["city"]

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("location", "city")


# --- Skill ---


def test_empty_skill_name_is_rejected():
    payload = valid_payload()
    payload["skills"][0]["name"] = ""

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    assert exc_info.value.errors()[0]["type"] == "string_too_short"


# --- Requirement ---


@pytest.mark.parametrize("field", ["description", "source_text"])
def test_blank_requirement_text_is_rejected(field):
    payload = valid_payload()
    payload["requirements"][0][field] = ""

    with pytest.raises(ValidationError) as exc_info:
        JobAnalysis.model_validate(payload)

    assert exc_info.value.errors()[0]["type"] == "string_too_short"


@pytest.mark.parametrize(
    "kind, importance, expected",
    [
        ("eligibility", "required", True),
        ("eligibility", "preferred", False),
        ("education", "required", False),
        ("language", "required", False),
        ("other", "required", False),
    ],
)
def test_hard_gate_rule(kind, importance, expected):
    requirement = Requirement(
        kind=kind, importance=importance, description="x", source_text="x"
    )

    assert requirement.is_hard_gate is expected


def test_hard_gate_is_not_part_of_the_schema():
    # The LLM must never be asked to decide what is a hard gate.
    assert "is_hard_gate" not in str(JobAnalysis.model_json_schema())
