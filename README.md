# 📰 News Headline Generator — The Headline Desk

A production-style news headline generator that uses **two parallel backends**: a **fine-tuned local model** (flan-t5-small + LoRA adapter, trained via Google Colab) and/or the **Groq API** (llama-3.3-70b-versatile). Headlines are **validated post-generation** — deduplicated, length-checked, and grounded against the source article — before they reach the UI.

Built as a **Streamlit web app** with a vintage press-dispatch visual identity, plus a CLI for quick terminal use.

---

## ✨ Features

- **Dual-backend headline generation**
  - **Fine-tuned local model** — A PEFT/LoRA adapter trained on flan-t5-small in Colab, runs entirely on your machine (CPU or GPU).
  - **Groq API** — Uses `llama-3.3-70b-versatile` for comparison, style-aware generation (neutral/clickbait/SEO).
- **Post-generation validation pipeline**
  - **Deduplication** — removes exact and near-duplicate headlines.
  - **Length enforcement** — drops headlines over the configured `max_chars` limit.
  - **Grounding check** — ensures any numbers or proper nouns in the headline appear in the source article (regex-based proxy for hallucination detection).
- **Regeneration support** — clicking "Generate" again produces fresh headlines (sampling-based diversity + generation counter).
- **Length control** — both backends respect the same `max_chars` slider; Groq is forcefully prompted to fill the character budget (80%–100% of max).
- **Article scraping** — fetch article text from a URL via `trafilatura` (with robots.txt compliance).
- **Comparison mode** — run both backends side-by-side in the UI to evaluate fine-tuned vs. API performance.
- **Debug tools** — optional expanders for raw model output and detailed validation warnings.
- **Vintage press aesthetic** — wire-ticket headline cards, telegram-slip input form, red "TRANSMIT" button.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.38+ |
| **Local Model** | HuggingFace `transformers`, `peft`, `torch`, `flan-t5-small` + LoRA adapter |
| **API Backend** | Groq SDK (`llama-3.3-70b-versatile`) |
| **Scraping** | `trafilatura`, `requests`, `urllib.robotparser` |
| **Validation** | `pydantic` schemas, custom regex grounding check |
| **Packaging** | `uv` (fast Python package manager) |
| **Runtime** | Python ≥ 3.11 |

---

## 📁 Project Structure

```
news_headline/
├── app.py                              # Streamlit web UI — the main user interface
├── pyproject.toml                       # Project metadata & dependencies (uv)
├── uv.lock                             # Locked dependency versions
├── .gitignore
├── README.md                           # ← this file
│
├── apps/                               # Core application modules
│   ├── cli.py                          # Terminal-based CLI entry point
│   ├── groq_client.py                  # Groq API client + optional local model switch
│   ├── local_model_client.py           # Fine-tuned flan-t5-small + LoRA loader & generator
│   ├── pipeline.py                     # Orchestrator: input → generate → validate → output
│   ├── schemas.py                      # Pydantic models (ArticleInput, Headline, HeadlineResponse)
│   ├── validators.py                   # Dedup, length filter, grounding check logic
│   └── scraper.py                      # URL fetching & article extraction (trafilatura)
│
├── Tuning-in-colab/                    # Colab training artifacts
│   └── headline-lora-adapter/          # LoRA adapter weights & config
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── README.md
│
└── tests/
    └── test_tuned_backend.py           # Unit tests for the local model backend
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Environment setup

Copy the example env file and add your **Groq API key** (free at [console.groq.com](https://console.groq.com)):

```bash
# Minimal — Groq only
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

For the fine-tuned local model (see "Colab training" below):

```env
USE_TUNED_MODEL=true
LOCAL_TUNED_ADAPTER_DIR=./Tuning-in-colab/headline-lora-adapter
LORA_ADAPTER_PATH=./Tuning-in-colab/headline-lora-adapter
```

### 3. Run the web UI

```bash
uv run streamlit run app.py
```

### 4. Run the CLI

```bash
uv run python -m apps.cli
```

---

## 🎯 How It Works

### Generation flow (pipeline.py)

```
User input (article text + settings)
              ↓
    ┌──────────────────┐
    │  ArticleInput    │  ← validated via Pydantic (min 20 chars)
    └──────────────────┘
              ↓
    ┌──────────────────┐
    │  Generate raw    │  ← local model OR Groq API (or both in comparison mode)
    │  candidates      │
    └──────────────────┘
              ↓
    ┌──────────────────┐
    │  Validate        │  ← dedupe → length filter → grounding check
    │                  │     (retries up to 2 times if too few pass)
    └──────────────────┘
              ↓
    ┌──────────────────┐
    │  HeadlineResponse│  ← structured output to UI/CLI
    └──────────────────┘
```

### Diversity & regeneration

- **Local model**: Uses `do_sample=True` with `temperature=0.85`, `top_k=50`, `top_p=0.92` for varied candidates. Each click increments a `generation_id` that seeds `torch.manual_seed()` for fresh output.
- **Groq**: Uses `temperature=0.8` and injects a variation seed into the prompt to request different headlines on regeneration.
- Previously used deterministic beam search → caused near-identical headlines differing only by articles ("a" vs "the").

### Length control

- **Single slider** (`max_chars`, 30–120 chars) controls both backends.
- **Groq prompt**: Uses strong directives — `"MUST be between {min_chars} and {max_chars}"` with 80% floor and "CRITICAL INSTRUCTION" emphasis — instead of the old weak "aim for approximately N" phrasing.
- **Local model**: Token budget derived from `max_chars // 4` (approximate chars/token ratio), enforced post-hoc by the length filter.

### Grounding check (hallucination prevention)

- `validators.py` extracts numbers and capitalized words from the headline and checks they exist in the source article.
- Headlines with unsupported facts are **dropped**, not just flagged — preventing fabricated entities from reaching the UI.
- Fallback: if all candidates fail grounding, the best-effort original is shown with a warning.

---

## 🧪 Colab Training (for the local adapter)

The fine-tuned model was trained in Google Colab using:

- **Base model**: `google/flan-t5-small` (~80M params)
- **Method**: LoRA (PEFT) — low-rank adapters, not full fine-tuning
- **Hardware**: Free T4 GPU
- **Dataset**: News article–headline pairs

The adapter weights live in `Tuning-in-colab/headline-lora-adapter/`. To train your own:

1. Open the Colab notebook (see `/Tuning-in-colab/`).
2. Replace the dataset with your own headline pairs.
3. Export the adapter to `headline-lora-adapter/`.
4. Point `LORA_ADAPTER_PATH` to the new folder.

---

## 🖥️ UI Controls

| Control | Description |
|---------|-------------|
| **Number of headlines** | 1–10 candidates per generation |
| **Max characters** | 30–120 character limit for all headlines |
| **Comparison mode** | Runs both Groq + local model side-by-side |
| **Show raw output** | Debug expander showing unvalidated model output |
| **Show detailed warnings** | Per-headline validation reasons instead of collapsed counts |
| **Fetch from URL** | Scrapes article text from a URL |
| **Clear fetched article** | Resets the scraped text |

---

## ⚠️ Known Limitations

- **Grounding check** uses simple regex (capitalized words + numbers), not real NER. May produce false positives/negatives. Swap in spaCy for stronger hallucination detection.
- **Local model** (flan-t5-small) is tiny — quality ceiling is lower than the Groq API. The LoRA adapter was trained on a limited dataset and may not generalize well to all article types.
- **Groq API** requires an internet connection and a free API key. The prompt engineering for length control is heuristic — results may vary.
- **Scraping** may fail on JS-rendered or paywalled pages.

---

## 📄 License

This project is for educational/demo purposes. See `pyproject.toml` for dependency licenses.
