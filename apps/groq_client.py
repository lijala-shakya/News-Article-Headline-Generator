"""Generation client that supports both Groq and a locally loaded tuned adapter."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

_client: Any | None = None
_local_model: Any | None = None
_local_tokenizer: Any | None = None


def should_use_local_model() -> bool:
    """Enable the local tuned adapter when explicitly requested."""
    return os.getenv("USE_TUNED_MODEL", "false").lower() in {"1", "true", "yes", "on"}


def get_client() -> Any:
    global _client
    if _client is None:
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("The groq package is required for the Groq backend.") from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Create a .env file with "
                "GROQ_API_KEY=your_key_here."
            )
        _client = Groq(api_key=api_key)
    return _client


def _load_local_tuned_model() -> tuple[Any, Any]:
    global _local_model, _local_tokenizer
    if _local_model is not None and _local_tokenizer is not None:
        return _local_model, _local_tokenizer

    adapter_dir = os.getenv("LOCAL_TUNED_ADAPTER_DIR")
    if not adapter_dir:
        adapter_dir = str(Path(__file__).resolve().parents[1] / "Tuning-in-colab" / "headline-lora-adapter")

    if not Path(adapter_dir).exists():
        raise RuntimeError(f"Local tuned adapter directory not found: {adapter_dir}")

    try:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "The local tuned-model backend requires 'torch', 'transformers', and 'peft'."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    _local_model = model
    _local_tokenizer = tokenizer
    return _local_model, _local_tokenizer


STYLE_INSTRUCTIONS = {
    "neutral": "Write factual, neutral, newspaper-style headlines. No sensationalism.",
    "clickbait": (
        "Write attention-grabbing, curiosity-driven headlines. Still must be "
        "factually accurate -- do not invent details."
    ),
    "seo": (
        "Write headlines optimized for search engines: front-load key terms, "
        "keep them clear and specific, avoid vague phrasing."
    ),
}


def _generate_with_local_model(article_text: str, style: str, num_candidates: int, max_chars: int) -> list[str]:
    model, tokenizer = _load_local_tuned_model()

    prompt = (
        f"Write {num_candidates} headline candidates for this article in a {style} style. "
        f"Keep each headline under {max_chars} characters.\n\nArticle:\n{article_text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=120, num_beams=4, do_sample=True, temperature=0.8, trust_remote_code=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not generated_text.strip():
        raise RuntimeError("The local tuned model returned no output.")

    return [line.strip(" -") for line in generated_text.splitlines() if line.strip()][:num_candidates]


def generate_raw_headlines(
    article_text: str,
    style: str,
    num_candidates: int,
    max_chars: int,
    min_length_ratio: float = 0.0,
    seed: int | None = None,
) -> list[str]:
    """Generate headlines using the tuned local model when enabled, otherwise fall back to Groq.

    min_length_ratio, if > 0, is used to compute a minimum character target
    for each headline (floor = max_chars * min_length_ratio) so the model
    doesn't default to short fragments that technically fit under max_chars
    but are too vague to be useful.

    seed: optional seed/identifier used to request varied output on regeneration.
    """
    if should_use_local_model():
        try:
            return _generate_with_local_model(article_text, style, num_candidates, max_chars)
        except Exception as exc:
            raise RuntimeError(f"Local tuned model failed: {exc}") from exc

    client = get_client()

    # Force Groq to fill the requested character budget. Previously the prompt
    # said "aim for approximately N characters" which was too weak -- the model
    # consistently stayed at 50-70 chars regardless of max_chars. Now we set a
    # strict minimum floor at 80% of max_chars and use strong directives.
    min_chars = int(max_chars * 0.8)

    system_prompt = (
        "You are a professional news headline writer. "
        f"{STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS['neutral'])}\n"
        f"Each headline MUST be between {min_chars} and {max_chars} characters long. "
        f"This is a strict requirement -- do not write headlines shorter than "
        f"{min_chars} characters or longer than {max_chars} characters."
    )

    system_prompt += (
        "\nCRITICAL INSTRUCTION: Write every headline as close to "
        f"{max_chars} characters as possible. Use the full character budget "
        "to be specific, detailed and informative. Short vague headlines "
        "are unacceptable. Be descriptive and comprehensive."
    )

    system_prompt += (
        "\nEvery headline must be fully supported by the article text -- do not "
        "add names, numbers, or claims that are not in the source.\n"
        "Respond ONLY with a JSON object in this exact shape, no other text:\n"
        '{"headlines": ["headline 1", "headline 2", ...]}'
    )

    user_prompt = (
        f"Generate exactly {num_candidates} distinct headline candidates for "
        f"this article. Remember: each headline MUST be between {min_chars} "
        f"and {max_chars} characters -- aim for around {max_chars}:\n\n"
        f"{article_text}"
    )

    # Add seed to the user prompt to vary output on regeneration
    if seed is not None:
        user_prompt += f"\n\n(Variation seed: {seed})"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    headlines = parsed.get("headlines", [])

    if not isinstance(headlines, list) or not headlines:
        raise ValueError(f"Model did not return a valid headline list: {raw!r}")

    return [str(h) for h in headlines]
