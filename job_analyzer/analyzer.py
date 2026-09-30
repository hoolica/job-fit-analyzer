"""Business logic between the CLI and the LLM client.

Extraction is delegated to an injectable extractor; this module only cleans
the result. Type and business rules stay in models.py: the cleaned data is
rebuilt through JobAnalysis.model_validate, never re-checked here.
"""

from collections.abc import Callable

from job_analyzer.llm_client import extract_job_analysis
from job_analyzer.models import JobAnalysis

Extractor = Callable[[str], JobAnalysis]


def analyze_job_offer(
    job_text: str,
    *,
    extract: Extractor = extract_job_analysis,
) -> JobAnalysis:
    """Extract a JobAnalysis from raw text, then normalize and deduplicate it."""
    return normalize_job_analysis(extract(job_text))


def normalize_job_analysis(analysis: JobAnalysis) -> JobAnalysis:
    """Return a cleaned copy of analysis. Pure function, no I/O."""
    data = analysis.model_dump()

    data["job_title"] = _clean_text(data["job_title"])
    data["company"] = _clean_text(data["company"])
    data["skills"] = _deduplicate_skills(data["skills"])
    data["responsibilities"] = _clean_list(data["responsibilities"])
    data["benefits"] = _clean_list(data["benefits"])
    data["requirements"] = _deduplicate_requirements(data["requirements"])

    experience = data["experience"]
    experience["source_text"] = _strip_or_none(experience["source_text"])

    location = data["location"]
    location["city"] = _clean_text(location["city"])
    location["country"] = _clean_text(location["country"])

    salary = data["salary"]
    currency = _clean_text(salary["currency"])
    salary["currency"] = currency.upper() if currency else None
    salary["source_text"] = _strip_or_none(salary["source_text"])

    return JobAnalysis.model_validate(data)


def _clean_text(value: str | None) -> str | None:
    """Trim and collapse inner whitespace; blank becomes None."""
    if value is None:
        return None
    return " ".join(value.split()) or None


def _strip_or_none(value: str | None) -> str | None:
    """Trim the ends only, keeping a verbatim excerpt intact; blank becomes None."""
    if value is None:
        return None
    return value.strip() or None


def _clean_list(items: list[str]) -> list[str]:
    """Clean each item, drop blanks and case-insensitive duplicates, keep order."""
    seen: set[str] = set()
    result = []
    for item in items:
        cleaned = _clean_text(item)
        if cleaned is None or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        result.append(cleaned)
    return result


def _deduplicate_skills(skills: list[dict]) -> list[dict]:
    """Merge skills sharing a case-insensitive name.

    The first occurrence keeps its position, spelling and category;
    required is True if any occurrence is required.
    """
    merged: dict[str, dict] = {}
    for skill in skills:
        name = _clean_text(skill["name"])
        if name is None:
            continue
        key = name.casefold()
        if key in merged:
            merged[key]["required"] = merged[key]["required"] or skill["required"]
        else:
            merged[key] = {**skill, "name": name}
    return list(merged.values())


def _deduplicate_requirements(requirements: list[dict]) -> list[dict]:
    """Clean requirements and merge those sharing kind and description.

    A requirement whose description or source_text is blank is dropped.
    The first occurrence keeps its position, wording and source_text;
    importance is "required" if any occurrence is required.
    """
    merged: dict[tuple[str, str], dict] = {}
    for requirement in requirements:
        description = _clean_text(requirement["description"])
        source_text = _strip_or_none(requirement["source_text"])
        if description is None or source_text is None:
            continue
        key = (requirement["kind"], description.casefold())
        if key in merged:
            if requirement["importance"] == "required":
                merged[key]["importance"] = "required"
        else:
            merged[key] = {**requirement, "description": description, "source_text": source_text}
    return list(merged.values())
