"""Orchestrates the full headline generation pipeline:
input validation -> API call -> post-validation -> structured output.
"""

from apps.schemas import ArticleInput, HeadlineResponse
from apps.groq_client import generate_raw_headlines
from apps.validators import build_validated_headlines

MIN_ARTICLE_CHARS = 20
MAX_REGENERATE_ATTEMPTS = 2


def generate_headlines(
    article_text: str,
    style: str = "neutral",
    num_candidates: int = 3,
    max_chars: int = 70,
    min_length_ratio: float = 0.6,
    seed: int | None = None,
    temperature: float = 0.8,
) -> HeadlineResponse:
    """
    Main entry point. Validates input, calls the model, validates output,
    and retries once if nothing survived validation.

    min_length_ratio is passed through to the Groq prompt as a target floor
    (see groq_client.generate_raw_headlines) so headlines aren't just short
    fragments that happen to fit under max_chars.

    seed: optional seed for varied output on regeneration. Pass a different
    value each time you click generate to get fresh headlines for the same article.

    temperature: controls randomness in Groq's sampling. Higher values (e.g. 1.2)
    produce more diverse headlines; lower values (e.g. 0.3) more conservative.
    Range 0.1–2.0. Default 0.8.
    """
    article = ArticleInput(
        text=article_text,
        style=style,
        num_candidates=num_candidates,
        max_chars=max_chars,
    )

    all_warnings: list[str] = []
    validated = []

    for attempt in range(MAX_REGENERATE_ATTEMPTS):
        raw = generate_raw_headlines(
            article_text=article.text,
            style=article.style.value,
            num_candidates=article.num_candidates,
            max_chars=article.max_chars,
            min_length_ratio=min_length_ratio,
            seed=seed,
            temperature=temperature,
        )
        validated, warnings = build_validated_headlines(
            raw_headlines=raw,
            article_text=article.text,
            max_chars=article.max_chars,
            num_candidates=article.num_candidates,
        )
        all_warnings.extend(warnings)

        if len(validated) >= article.num_candidates:
            break
        if attempt < MAX_REGENERATE_ATTEMPTS - 1:
            all_warnings.append("Regenerating: not enough valid candidates on first pass.")

    return HeadlineResponse(
        style=article.style,
        candidates=validated,
        source_char_count=len(article.text),
        warnings=all_warnings,
        raw_candidates=raw,
    )


def generate_headlines_local(
    article_text: str,
    num_candidates: int = 3,
    max_chars: int = 70,
    seed: int | None = None,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> HeadlineResponse:
    """
    Primary entry point using ONLY the fine-tuned local model -- no API call,
    no network dependency, runs entirely on your machine.

    Runs the same dedup/length/grounding validation as the Groq path, so
    output quality checks stay consistent regardless of which model produced
    the candidates.

    Note: 'style' isn't passed to the model (it wasn't trained on styles),
    it's kept only so callers can label the output consistently with the
    Groq path.

    seed: optional seed for varied output on regeneration. Pass a different
    value each time you click generate to get fresh headlines for the same article.

    temperature: controls randomness in local model sampling. Range 0.1–2.0. Default 0.85.
    top_p: nucleus sampling threshold. Range 0.1–1.0. Default 0.92.
    """
    article = ArticleInput(
        text=article_text,
        style="neutral",
        num_candidates=num_candidates,
        max_chars=max_chars,
    )

    from apps.local_model_client import generate_local_headlines

    all_warnings: list[str] = []
    validated = []

    for attempt in range(MAX_REGENERATE_ATTEMPTS):
        raw = generate_local_headlines(
            article_text=article.text,
            num_candidates=article.num_candidates,
            max_chars=article.max_chars,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )
        validated, warnings = build_validated_headlines(
            raw_headlines=raw,
            article_text=article.text,
            max_chars=article.max_chars,
            num_candidates=article.num_candidates,
        )
        all_warnings.extend(warnings)

        if len(validated) >= article.num_candidates:
            break
        if attempt < MAX_REGENERATE_ATTEMPTS - 1:
            all_warnings.append("Regenerating: not enough valid candidates on first pass.")

    if not validated:
        all_warnings.append(
            "Fine-tuned model produced no candidates passing validation "
            "(common with a lightly-trained small model) -- showing raw output instead."
        )
        validated, _ = build_validated_headlines(
            raw_headlines=raw,
            article_text=article.text,
            max_chars=max_chars,
            num_candidates=article.num_candidates,
        )

    return HeadlineResponse(
        style=article.style,
        candidates=validated,
        source_char_count=len(article.text),
        warnings=all_warnings,
        raw_candidates=raw,
    )


def generate_headlines_with_comparison(
    article_text: str,
    style: str = "neutral",
    num_candidates: int = 3,
    max_chars: int = 70,
    min_length_ratio: float = 0.6,
    seed: int | None = None,
    temperature_groq: float = 0.8,
    temperature_local: float = 0.85,
    top_p_local: float = 0.92,
) -> dict:
    """
    Runs the Groq pipeline AND the local fine-tuned model on the same article,
    for side-by-side comparison. This is the 'Task 2 stretch goal in action'
    entry point -- use this in the UI's comparison view / your writeup.

    Splits the requested `num_candidates` between the two backends so the
    COMBINED total matches what the user asked for -- e.g. asking for 4
    gives 2 from Groq + 2 from the local model, not 4 from each. Groq gets
    the extra one on odd counts since it's the more reliable backend and
    less likely to lose candidates to validation.

    Returns a dict rather than a strict schema since the two sources have
    different shapes (Groq gives multiple candidates, the local model gives one).
    Local model errors are caught and surfaced as a message rather than crashing
    the whole comparison, since it's optional and the adapter may not be present.

    seed: optional seed for varied output on regeneration. Pass a different
    value each time you click generate to get fresh headlines for the same article.
    """
    groq_count = (num_candidates + 1) // 2
    local_count = num_candidates - groq_count

    groq_result = generate_headlines(
        article_text=article_text,
        style=style,
        num_candidates=groq_count,
        max_chars=max_chars,
        min_length_ratio=min_length_ratio,
        seed=seed,
        temperature=temperature_groq,
    )

    try:
        if local_count > 0:
            # Request extra raw candidates so validation filtering doesn't
            # leave us with fewer headlines than requested.
            local_raw_count = local_count * 3
            local_result = generate_headlines_local(
                article_text=article_text,
                num_candidates=local_raw_count,
                max_chars=max_chars,
                seed=seed,
                temperature=temperature_local,
                top_p=top_p_local,
            )
            local_headlines = [h.text for h in local_result.candidates][:local_count]
            local_warnings = local_result.warnings
            local_raw = local_result.raw_candidates
        else:
            local_headlines = []
            local_warnings = []
            local_raw = []
        local_error = None
    except Exception as e:
        local_headlines = []
        local_warnings = []
        local_raw = []
        local_error = str(e)

    return {
        "groq_candidates": [h.text for h in groq_result.candidates],
        "groq_warnings": groq_result.warnings,
        "groq_raw_candidates": groq_result.raw_candidates,
        "local_finetuned_headlines": local_headlines,
        "local_warnings": local_warnings,
        "local_raw_candidates": local_raw,
        "local_error": local_error,
    }
