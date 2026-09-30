import pytest

from job_analyzer.analyzer import analyze_job_offer, normalize_job_analysis
from job_analyzer.llm_client import LLMRefusalError
from job_analyzer.models import JobAnalysis


def clean_payload() -> dict:
    """An analysis that is already normalized."""
    return {
        "job_title": "iOS Developer",
        "company": "McKinsey",
        "contract_type": "permanent",
        "skills": [
            {"name": "Swift", "category": "language", "required": True},
            {"name": "SQL", "category": "technical", "required": False},
        ],
        "responsibilities": ["Build the mobile app"],
        "experience": {
            "min_years": 2,
            "max_years": None,
            "level": "mid",
            "source_text": "2+ years",
        },
        "location": {"city": "Montréal", "country": "Canada", "work_mode": "hybrid"},
        "salary": {
            "min_amount": 75000,
            "max_amount": 90000,
            "currency": "CAD",
            "period": "yearly",
            "source_text": "75 000 $ - 90 000 $",
        },
        "benefits": ["Remote Fridays"],
        "requirements": [
            {
                "kind": "language",
                "description": "English",
                "importance": "required",
                "source_text": "Knowledge of English is required",
            },
        ],
    }


def make_analysis(**changes) -> JobAnalysis:
    """Build a JobAnalysis from clean_payload(), overriding nested paths.

    Keys use "__" as a separator: make_analysis(location__city=" Paris ").
    """
    data = clean_payload()
    for path, value in changes.items():
        *parents, last = path.split("__")
        target = data
        for key in parents:
            target = target[key]
        target[last] = value
    return JobAnalysis.model_validate(data)


def get_at(analysis: JobAnalysis, path: str):
    value = analysis.model_dump()
    for key in path.split("__"):
        value = value[key]
    return value


def skill(name: str, required: bool = True, category: str = "technical") -> dict:
    return {"name": name, "category": category, "required": required}


# --- analyze_job_offer ---


def test_analyze_passes_raw_text_to_extractor_and_normalizes(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    received = []

    def fake_extract(text: str) -> JobAnalysis:
        received.append(text)
        return make_analysis(
            job_title="  iOS   Developer\n",
            skills=[skill("Swift"), skill("swift", required=False)],
        )

    result = analyze_job_offer("  raw offer text\n", extract=fake_extract)

    assert received == ["  raw offer text\n"]
    assert result.job_title == "iOS Developer"
    assert [s.name for s in result.skills] == ["Swift"]


def test_extractor_errors_propagate_unchanged():
    error = LLMRefusalError("declined")

    def failing_extract(text: str) -> JobAnalysis:
        raise error

    with pytest.raises(LLMRefusalError) as exc_info:
        analyze_job_offer("offer", extract=failing_extract)

    assert exc_info.value is error


# --- normalize_job_analysis: text fields ---


@pytest.mark.parametrize(
    "path",
    ["job_title", "company", "location__city", "location__country"],
)
def test_single_line_text_is_trimmed_and_collapsed(path):
    result = normalize_job_analysis(make_analysis(**{path: "  New \n  York\t"}))

    assert get_at(result, path) == "New York"


@pytest.mark.parametrize(
    "path",
    [
        "job_title",
        "company",
        "location__city",
        "location__country",
        "salary__currency",
        "experience__source_text",
        "salary__source_text",
    ],
)
def test_blank_text_becomes_none(path):
    result = normalize_job_analysis(make_analysis(**{path: " \n\t "}))

    assert get_at(result, path) is None


def test_currency_is_trimmed_and_uppercased():
    result = normalize_job_analysis(make_analysis(salary__currency=" cad "))

    assert result.salary.currency == "CAD"


def test_source_text_keeps_its_inner_formatting():
    excerpt = "Salary:\n75k  -  90k"
    result = normalize_job_analysis(make_analysis(salary__source_text=f"  {excerpt}\n"))

    assert result.salary.source_text == excerpt


@pytest.mark.parametrize("field", ["responsibilities", "benefits"])
def test_text_lists_are_cleaned_and_deduplicated(field):
    items = ["  Write  tests ", "", "   ", "write tests", "Review code"]

    result = normalize_job_analysis(make_analysis(**{field: items}))

    assert getattr(result, field) == ["Write tests", "Review code"]


# --- normalize_job_analysis: what must not change ---


def test_clean_analysis_is_left_unchanged():
    analysis = make_analysis()

    assert normalize_job_analysis(analysis) == analysis


def test_input_is_not_mutated():
    analysis = make_analysis(job_title="  iOS  Developer ", skills=[skill("SQL"), skill("sql")])
    before = analysis.model_dump()

    normalize_job_analysis(analysis)

    assert analysis.model_dump() == before


def test_normalization_is_idempotent():
    messy = make_analysis(
        job_title=" iOS   Developer ",
        salary__currency="cad",
        skills=[skill(" Swift "), skill("SWIFT", required=False)],
        benefits=["A", "a", " "],
    )

    once = normalize_job_analysis(messy)

    assert normalize_job_analysis(once) == once


# --- Skill deduplication ---


def test_duplicate_skills_keep_first_spelling_category_and_position():
    skills = [
        skill("Docker", category="tool"),
        skill("  python ", category="language"),
        skill("Git", category="tool"),
        skill("PYTHON", category="technical"),
    ]

    result = normalize_job_analysis(make_analysis(skills=skills))

    assert [(s.name, s.category) for s in result.skills] == [
        ("Docker", "tool"),
        ("python", "language"),
        ("Git", "tool"),
    ]


@pytest.mark.parametrize(
    "first_required, second_required, expected",
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_duplicate_skills_required_true_wins(first_required, second_required, expected):
    skills = [skill("Python", first_required), skill("python", second_required)]

    result = normalize_job_analysis(make_analysis(skills=skills))

    assert len(result.skills) == 1
    assert result.skills[0].required is expected


def test_aliases_are_not_merged():
    skills = [skill("JS"), skill("JavaScript"), skill("React"), skill("React.js")]

    result = normalize_job_analysis(make_analysis(skills=skills))

    assert [s.name for s in result.skills] == ["JS", "JavaScript", "React", "React.js"]


def test_skill_with_blank_name_is_dropped():
    skills = [skill("   "), skill("Python")]

    result = normalize_job_analysis(make_analysis(skills=skills))

    assert [s.name for s in result.skills] == ["Python"]


# --- Requirement normalization ---


def requirement(
    description: str, importance: str = "required", kind: str = "education", source_text: str = "quote"
) -> dict:
    return {
        "kind": kind,
        "description": description,
        "importance": importance,
        "source_text": source_text,
    }


def test_requirement_text_is_cleaned():
    raw = requirement("  Degree  in\nComputer Science ", source_text="  Degree in\n  CS  ")

    result = normalize_job_analysis(make_analysis(requirements=[raw]))

    assert result.requirements[0].description == "Degree in Computer Science"
    assert result.requirements[0].source_text == "Degree in\n  CS"


@pytest.mark.parametrize("field", ["description", "source_text"])
def test_requirement_with_blank_text_is_dropped(field):
    blank = requirement("English", kind="language")
    blank[field] = "   "

    result = normalize_job_analysis(make_analysis(requirements=[blank, requirement("Degree")]))

    assert [r.description for r in result.requirements] == ["Degree"]


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ("required", "preferred", "required"),
        ("preferred", "required", "required"),
        ("preferred", "preferred", "preferred"),
    ],
)
def test_duplicate_requirements_required_wins(first, second, expected):
    requirements = [
        requirement("Degree in CS", first, source_text="first quote"),
        requirement("degree in cs", second, source_text="second quote"),
    ]

    result = normalize_job_analysis(make_analysis(requirements=requirements))

    [merged] = result.requirements
    assert merged.importance == expected
    assert merged.description == "Degree in CS"
    assert merged.source_text == "first quote"


def test_requirements_of_different_kinds_are_not_merged():
    requirements = [requirement("English", kind="language"), requirement("English", kind="other")]

    result = normalize_job_analysis(make_analysis(requirements=requirements))

    assert [r.kind for r in result.requirements] == ["language", "other"]
