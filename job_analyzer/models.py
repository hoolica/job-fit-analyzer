"""Data contract for job offer analysis.

Nullable fields have no default on purpose: they must always be present
in the payload (and in the JSON schema), but may explicitly be null.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Skill(StrictModel):
    name: str = Field(min_length=1)
    category: Literal["technical", "tool", "language", "soft_skill"]
    required: bool


class Experience(StrictModel):
    min_years: int | None = Field(ge=0)
    max_years: int | None = Field(ge=0)
    level: Literal["internship", "junior", "mid", "senior", "lead"] | None
    source_text: str | None

    @model_validator(mode="after")
    def check_years_range(self) -> "Experience":
        if (
            self.min_years is not None
            and self.max_years is not None
            and self.min_years > self.max_years
        ):
            raise ValueError("min_years must be less than or equal to max_years")
        return self


class Location(StrictModel):
    city: str | None
    country: str | None
    work_mode: Literal["on_site", "hybrid", "remote"] | None


class Salary(StrictModel):
    min_amount: int | None = Field(ge=0)
    max_amount: int | None = Field(ge=0)
    currency: str | None
    period: Literal["yearly", "monthly", "daily", "hourly"] | None
    source_text: str | None

    @model_validator(mode="after")
    def check_amount_range(self) -> "Salary":
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("min_amount must be less than or equal to max_amount")
        return self


class JobAnalysis(StrictModel):
    job_title: str | None
    company: str | None
    contract_type: Literal[
        "permanent", "fixed_term", "internship", "apprenticeship", "freelance"
    ] | None
    skills: list[Skill]
    responsibilities: list[str]
    experience: Experience
    location: Location
    salary: Salary
    benefits: list[str]
