"""Validation & post-processing for generated headlines.

Kept dependency-light on purpose: uses simple regex/set-overlap heuristics
rather than a full NER model, so it runs fast on a laptop with no GPU.
For a stronger hallucination check later, swap `extract_numbers_and_caps`
for a spaCy NER pass.
"""

import re

from apps.schemas import Headline


def _dedupe(headlines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for h in headlines:
        key = h.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(h.strip())
    return unique


def _truncate_over_limit(headlines: list[str], max_chars: int) -> list[str]:
    """Drop headlines that exceed max_chars rather than silently truncating,
    since a truncated headline can cut off mid-word or change meaning."""
    return [h for h in headlines if len(h) <= max_chars]


def extract_numbers_and_caps(text: str) -> set[str]:
    """Pulls out numbers and capitalized words/phrases as a cheap proxy for
    'facts that must be grounded in the source article'."""
    numbers = set(re.findall(r"\b\d[\d,\.]*\b", text))
    # capitalized words, excluding the very first word of each sentence
    caps = set(re.findall(r"(?<!^)(?<!\. )\b[A-Z][a-zA-Z]+\b", text))
    return numbers | caps


def find_unsupported_facts(headline: str, article_text: str) -> set[str]:
    """Returns the set of numbers/proper-noun phrases in the headline that
    don't appear anywhere in the source article -- the raw signal used by
    both check_grounding (for warnings) and build_validated_headlines
    (to actually reject ungrounded headlines, not just flag them)."""
    article_facts = extract_numbers_and_caps(article_text)
    headline_facts = extract_numbers_and_caps(headline)

    return {
        fact for fact in headline_facts
        if fact not in article_facts and fact.lower() not in article_text.lower()
    }


def check_grounding(headline: str, article_text: str) -> list[str]:
    """Returns a list of warning strings if the headline introduces numbers
    or proper nouns not found anywhere in the source article."""
    unsupported = find_unsupported_facts(headline, article_text)

    warnings = []
    if unsupported:
        warnings.append(
            f"Possible unsupported detail in headline: {', '.join(sorted(unsupported))}"
        )
    return warnings


def build_validated_headlines(
    raw_headlines: list[str],
    article_text: str,
    max_chars: int,
    num_candidates: int,
) -> tuple[list[Headline], list[str]]:
    """
    Runs the full validation pipeline: dedupe -> length filter -> grounding check.
    Returns (validated_headline_objects, warnings).
    """
    warnings: list[str] = []

    deduped = _dedupe(raw_headlines)
    if len(deduped) < len(raw_headlines):
        warnings.append("Removed duplicate/near-duplicate headline candidates.")

    within_length = _truncate_over_limit(deduped, max_chars)
    if len(within_length) < len(deduped):
        warnings.append(
            f"Dropped {len(deduped) - len(within_length)} headline(s) exceeding "
            f"{max_chars} characters."
        )

    if not within_length:
        warnings.append(
            "No headlines survived validation — falling back to best available."
        )
        within_length = deduped[:1] if deduped else raw_headlines[:1]

    grounded: list[str] = []
    dropped_count = 0
    for h in within_length:
        unsupported = find_unsupported_facts(h, article_text)
        if unsupported:
            dropped_count += 1
            warnings.append(
                f"Dropped headline with unsupported detail "
                f"({', '.join(sorted(unsupported))}): \"{h}\""
            )
        else:
            grounded.append(h)

    if not grounded:

        warnings.append(
            "All candidates contained unsupported details — showing the "
            "original best-effort candidate, unverified."
        )
        grounded = within_length[:1]

    results: list[Headline] = [
        Headline(text=h, char_count=len(h)) for h in grounded[:num_candidates]
    ]

    return results, warnings