"""Loads the locally fine-tuned flan-t5-small + LoRA adapter and generates
headlines with it. Runs on CPU -- the model is tiny enough that this is fine
even without a GPU.

This is a SEPARATE path from groq_client.py, not a replacement. Use it for
side-by-side comparison against the Groq pipeline (see pipeline.py's
generate_headlines_with_comparison()).
"""

import os
import random
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "Tuning-in-colab" / "headline-lora-adapter"
ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(DEFAULT_ADAPTER_DIR))
BASE_MODEL_NAME = "google/flan-t5-small"
PREFIX = "generate headline: "

MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 32


@lru_cache(maxsize=1)
def _load_local_model():
    """Loads once and caches -- avoid reloading the model on every call."""
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"LoRA adapter not found at '{ADAPTER_PATH}'. Download the "
            f"'headline-lora-adapter' folder from your Colab session (or Drive) "
            f"and place it at that path, or set LORA_ADAPTER_PATH in your .env."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


def generate_local_headline(article_text: str) -> str:
    """Generates a single headline using the fine-tuned local model.
    No style parameter -- this model was only trained on one style.
    Kept for the comparison view; generate_local_headlines() below is the
    one used for standalone (non-comparison) generation."""
    tokenizer, model = _load_local_model()

    inputs = tokenizer(
        PREFIX + article_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=MAX_TARGET_LEN, num_beams=4,
            trust_remote_code=True,
        )
    headline = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return re.sub(r"\s+", " ", headline).strip()


def generate_local_headlines(
    article_text: str, num_candidates: int = 3, max_chars: int | None = None,
    seed: int | None = None,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> list[str]:
    """
    Generates multiple headline candidates from the fine-tuned model using
    temperature-based sampling with top-k and top-p filtering. This produces
    diverse candidates on every call by combining random sampling with beam
    search diversity.

    Note: unlike the Groq path, this model has no style conditioning -- all
    candidates come from the same single fine-tuned "voice" it learned.

    max_chars is optional and, if given, is converted into an approximate
    token budget for generation (rough estimate: ~3 characters/token for
    English). This doesn't guarantee every output is under max_chars --
    that's still enforced downstream in validators.py -- but it steers
    generation toward shorter output instead of relying entirely on the
    post-hoc length filter to drop everything.

    seed: used to seed PyTorch and Python random RNGs for varied output
    on re-generation. Passing different seeds ensures fresh headlines.

    temperature: controls randomness in sampling. Higher values (e.g. 1.2)
    produce more diverse headlines; lower values (e.g. 0.3) produce more
    conservative, predictable output. Range 0.1-2.0. Default 0.85.

    top_p: nucleus sampling threshold. 0.92 means the model samples from
    the smallest set of tokens whose cumulative probability >= 92%.
    Lower values (e.g. 0.5) make output more focused; higher values
    (e.g. 1.0) allow more variety. Range 0.1-1.0. Default 0.92.
    """
    tokenizer, model = _load_local_model()

    inputs = tokenizer(
        PREFIX + article_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )

    target_len = MAX_TARGET_LEN
    if max_chars is not None:
        target_len = max(10, min(MAX_TARGET_LEN, max_chars // 3))

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=target_len,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=top_p,
            trust_remote_code=True,
            num_return_sequences=num_candidates,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    headlines = [
        re.sub(r"\s+", " ", tokenizer.decode(ids, skip_special_tokens=True)).strip()
        for ids in output_ids
    ]
    return headlines
