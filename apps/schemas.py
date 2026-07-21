"""Pydantic models for the News Headline Generator."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class HeadlineStyle(str, Enum):
    NEUTRAL = "neutral"
    CLICKBAIT = "clickbait"
    SEO = "seo"


class ArticleInput(BaseModel):
    """Validated input article."""

    text: str = Field(..., min_length=20, description="Full article body text")
    style: HeadlineStyle = HeadlineStyle.NEUTRAL
    num_candidates: int = Field(default=3, ge=1, le=10)
    max_chars: int = Field(default=70, ge=20, le=140)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Article text cannot be empty or whitespace only.")
        return v.strip()


class Headline(BaseModel):
    """A single generated headline candidate."""

    text: str
    char_count: int

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip().strip('"')


class HeadlineResponse(BaseModel):
    """Final structured output returned to the user / UI."""

    style: HeadlineStyle
    candidates: list[Headline]
    source_char_count: int
    warnings: list[str] = Field(default_factory=list)
    raw_candidates: list[str] = Field(
        default_factory=list,
        description="Unvalidated model output from the final generation attempt, "
        "kept for debugging -- not filtered by dedup/length/grounding.",
    )