"""Streamlit UI for the News Headline Generator.
Run with: streamlit run app.py

Visual identity: a "press dispatch" -- the generation form reads as a
telegram submission slip, and each generated headline renders as a
torn wire ticket, echoing how headlines actually moved through a
newsroom before they hit print.
"""

import os
import random
from pathlib import Path

import streamlit as st

from apps.pipeline import (
    generate_headlines_local,
    generate_headlines_with_comparison,
)
from apps.scraper import scrape_article

st.set_page_config(page_title="The Wire — Headline Desk", page_icon="\u16D8", layout="centered")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#1B1B1B"
PAPER = "#EDEAE1"
PAPER_RAISED = "#F6F4EE"
WIRE_RED = "#C41E3A"
TELETYPE_TEAL = "#2B6777"
AGED_GOLD = "#B8923D"
RULE = "#C9C4B4"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Source Serif 4', Georgia, serif;
}}

.stApp {{
    background-color: {PAPER};
    background-image:
        repeating-linear-gradient(0deg, rgba(0,0,0,0.015) 0px, transparent 1px, transparent 2px),
        repeating-linear-gradient(90deg, rgba(0,0,0,0.015) 0px, transparent 1px, transparent 2px);
}}

.block-container {{
    padding-top: 2rem;
    max-width: 760px;
}}

/* ---- Masthead ---- */
.masthead {{
    text-align: center;
    border-bottom: 4px double {INK};
    padding-bottom: 0.6rem;
    margin-bottom: 0.3rem;
}}
.masthead .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: {WIRE_RED};
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}}
.masthead h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    letter-spacing: 0.04em;
    color: {INK};
    margin: 0;
    line-height: 1;
}}
.masthead .tagline {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: {INK};
    opacity: 0.65;
    margin-top: 0.35rem;
}}
.dateline {{
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {INK};
    opacity: 0.55;
    letter-spacing: 0.1em;
    margin: 0.5rem 0 1.6rem 0;
    text-transform: uppercase;
}}

/* ---- Section labels ---- */
.slip-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {TELETYPE_TEAL};
    font-weight: 500;
    margin: 1.1rem 0 0.35rem 0;
    border-bottom: 1px solid {RULE};
    padding-bottom: 0.25rem;
}}

/* ---- Inputs ---- */
.stTextArea textarea {{
    background-color: {PAPER_RAISED} !important;
    border: 1px solid {INK} !important;
    border-radius: 2px !important;
    font-family: 'Source Serif 4', serif !important;
    color: {INK} !important;
}}
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: {PAPER_RAISED} !important;
    border: 1px solid {INK} !important;
    border-radius: 2px !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{
    background: {TELETYPE_TEAL} !important;
}}
.stCheckbox label p {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}}

/* ---- Transmit button ---- */
.stButton > button {{
    background-color: {WIRE_RED} !important;
    color: {PAPER} !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100%;
    transition: background-color 0.15s ease;
}}
.stButton > button:hover {{
    background-color: #9e1830 !important;
    color: {PAPER} !important;
}}

/* ---- Wire ticket (headline result card) ---- */
.ticket {{
    position: relative;
    background: {PAPER_RAISED};
    border: 1px solid {INK};
    border-top: none;
    padding: 0.9rem 1.1rem 0.8rem 1.1rem;
    margin-bottom: 1rem;
}}
.ticket::before {{
    content: "";
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 6px;
    background-image: radial-gradient({PAPER} 40%, transparent 41%);
    background-size: 12px 12px;
    background-position: -2px -6px;
}}
.ticket-head {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.01em;
    color: {INK};
    line-height: 1.15;
    margin: 0.3rem 0 0.5rem 0;
}}
.ticket-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TELETYPE_TEAL};
}}
.stamp {{
    position: absolute;
    top: 0.6rem;
    right: 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: {WIRE_RED};
    border: 1.5px solid {WIRE_RED};
    border-radius: 3px;
    padding: 0.12rem 0.4rem;
    transform: rotate(4deg);
    opacity: 0.85;
}}

/* ---- Warnings as a caution notice ---- */
.caution {{
    border-left: 3px solid {AGED_GOLD};
    background: rgba(184, 146, 61, 0.08);
    padding: 0.6rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: {INK};
    margin-top: 0.3rem;
}}

/* ---- Column headers for comparison mode ---- */
.desk-label {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.05em;
    color: {INK};
    border-bottom: 2px solid {INK};
    padding-bottom: 0.2rem;
    margin-bottom: 0.6rem;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def render_warnings(warnings: list[str]) -> None:
    """Renders validation warnings as collapsed caution notices."""
    if not warnings:
        return

    unsupported_count = sum(1 for w in warnings if w.startswith("Dropped headline with unsupported detail"))
    length_count = sum(1 for w in warnings if w.startswith("Dropped ") and "exceeding" in w)
    other = [
        w for w in warnings
        if not w.startswith("Dropped headline with unsupported detail")
        and not (w.startswith("Dropped ") and "exceeding" in w)
    ]

    if unsupported_count:
        st.markdown(
            f'<div class="caution">Filtered {unsupported_count} headline(s) containing '
            f'details not found in the source article.</div>',
            unsafe_allow_html=True,
        )
    if length_count:
        st.markdown(
            f'<div class="caution">Filtered {length_count} headline(s) exceeding the character limit.</div>',
            unsafe_allow_html=True,
        )
    for w in other:
        st.markdown(f'<div class="caution">{w}</div>', unsafe_allow_html=True)


WIRE_CODES = ["AP-NPL", "RTR-14", "UPI-07", "TEL-22", "WIRE-9"]

st.markdown(
    f"""
    <div class="masthead">
        <div class="eyebrow">Est. Task 1 &middot; Groq Wire Service</div>
        <h1>THE HEADLINE DESK</h1>
        <div class="tagline">paste copy &middot; pick a wire &middot; transmit</div>
    <div class="dateline">Dispatch No. {random.choice(WIRE_CODES)}-{random.randint(1000,9999)}</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="slip-label">Transmission Settings</div>', unsafe_allow_html=True)

    compare_mode = st.checkbox(
        "Also compare against the Groq API",
        help="Optional -- requires GROQ_API_KEY in your .env.",
    )

    num_candidates = st.slider("Number of headlines", 1, 10, 4)
    max_chars = st.slider("Max characters", 30, 120, 70)

    st.markdown('<div class="slip-label">Sampling Controls</div>', unsafe_allow_html=True)

    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=2.0, value=0.8, step=0.05,
        help="Controls randomness. Higher = more diverse headlines; "
        "lower = more conservative/predictable.",
    )
    top_p = st.slider(
        "Top-p (nucleus sampling)",
        min_value=0.1, max_value=1.0, value=0.92, step=0.01,
        help="Nucleus sampling threshold. Lower = more focused output; "
        "higher = more variety.",
    )

    min_length_ratio = 0.0

    if not compare_mode:
        st.caption(
            "Note: the fine-tuned model wasn't trained with style options -- "
            "all candidates come from its one learned voice."
        )
    else:
        st.caption(
            "In comparison mode, the requested count is split between the "
            "two desks (Groq gets the extra one on odd numbers) so the "
            "combined total matches what you asked for."
        )

    style = "neutral"

    st.markdown('<div class="slip-label">Desk Status</div>', unsafe_allow_html=True)

    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    groq_dot = "\U0001F7E2" if groq_ready else "\u26AA"
    st.markdown(f"{groq_dot} Groq API key {'found' if groq_ready else 'not set'}")

    _default_adapter_dir = Path(__file__).resolve().parent / "Tuning-in-colab" / "headline-lora-adapter"
    _adapter_path = os.getenv("LORA_ADAPTER_PATH", str(_default_adapter_dir))
    local_ready = os.path.isdir(_adapter_path)
    local_dot = "\U0001F7E2" if local_ready else "\U0001F534"
    st.markdown(f"{local_dot} Local adapter {'found' if local_ready else 'missing'}")

    if compare_mode and not groq_ready:
        st.caption("Groq comparison will fail without a GROQ_API_KEY in your .env.")
    if not local_ready:
        st.caption(f"Expected local adapter at: {_adapter_path}")

    st.markdown('<div class="slip-label">About This Desk</div>', unsafe_allow_html=True)
    st.caption(
        "A production-style news headline generator that uses two parallel backends: a fine-tuned local model and/or the Groq API"
    )

    if st.session_state.get("scraped_text"):
        if st.button("Clear fetched article", use_container_width=True):
            st.session_state.scraped_text = ""
            st.rerun()

    st.markdown('<div class="slip-label">Performance</div>', unsafe_allow_html=True)
    try:
        import torch
        _device_note = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except ImportError:
        _device_note = "unknown"
    st.caption(f"Local model runs on: {_device_note}")

st.markdown('<div class="slip-label">Incoming Copy</div>', unsafe_allow_html=True)

if "scraped_text" not in st.session_state:
    st.session_state.scraped_text = ""

if "generation_id" not in st.session_state:
    st.session_state.generation_id = 0

url_col, fetch_col = st.columns([4, 1])
with url_col:
    url_input = st.text_input(
        "Or fetch from a URL",
        placeholder="https://example.com/some-article",
        label_visibility="collapsed",
    )
with fetch_col:
    fetch_clicked = st.button("Fetch \u2192")

if fetch_clicked:
    if not url_input.strip():
        st.error("Enter a URL to fetch first.")
    else:
        with st.spinner("Scraping article text..."):
            try:
                st.session_state.scraped_text = scrape_article(url_input.strip())
                st.success(f"Pulled {len(st.session_state.scraped_text)} characters from the wire.")
            except RuntimeError as e:
                st.error(f"Fetch failed: {e}")

article_text = st.text_area(
    "Article text",
    value=st.session_state.scraped_text,
    height=220,
    placeholder="Paste the full article body here, or fetch a URL above...",
    label_visibility="collapsed",
)

st.write("")
go = st.button("Transmit \u2192 Generate Headlines")

if go:
    if not article_text.strip():
        st.error("No copy on the wire yet — paste an article above first.")
    elif compare_mode:
        st.session_state.generation_id += 1
        with st.spinner("Wiring both desks..."):
            try:
                result = generate_headlines_with_comparison(
                    article_text=article_text,
                    style=style,
                    num_candidates=num_candidates,
                    max_chars=max_chars,
                    min_length_ratio=min_length_ratio,
                    seed=st.session_state.generation_id,
                    temperature_groq=temperature,
                    temperature_local=temperature,
                    top_p_local=top_p,
                )
            except Exception as e:
                st.error(f"Transmission failed: {e}")
                result = None

        if result:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="desk-label">Fine-Tuned Desk (Primary)</div>', unsafe_allow_html=True)
                if result["local_finetuned_headlines"]:
                    for i, h in enumerate(result["local_finetuned_headlines"], start=1):
                        st.markdown(
                            f"""<div class="ticket">
                                <div class="ticket-meta">CANDIDATE {i:02d} &middot; {len(h)} CHARS &middot; LOCAL MODEL</div>
                                <div class="ticket-head">{h}</div>""",
                            unsafe_allow_html=True,
                        )
                    if result["local_warnings"]:
                        with st.expander("Fine-tuned desk notes (validation warnings)"):
                            render_warnings(result["local_warnings"])
                else:
                    st.markdown(
                        f'<div class="caution">LOCAL WIRE DOWN — {result["local_error"]}</div>',
                        unsafe_allow_html=True,
                    )
            with col_b:
                st.markdown('<div class="desk-label">Groq Wire (Comparison)</div>', unsafe_allow_html=True)
                if result["groq_candidates"]:
                    for i, h in enumerate(result["groq_candidates"], start=1):
                        st.markdown(
                            f"""<div class="ticket">
                                <div class="ticket-meta">CANDIDATE {i:02d} &middot; {style.upper()}</div>
                                <div class="ticket-head">{h}</div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="caution">No Groq candidates survived validation.</div>',
                        unsafe_allow_html=True,
                    )
                if result["groq_warnings"]:
                    with st.expander("Groq desk notes (validation warnings)"):
                        render_warnings(result["groq_warnings"])
    else:
        st.session_state.generation_id += 1
        with st.spinner("Wiring the desk..."):
            try:
                result = generate_headlines_local(
                    article_text=article_text,
                    num_candidates=num_candidates,
                    max_chars=max_chars,
                    seed=st.session_state.generation_id,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                st.error(f"Transmission failed: {e}")
                result = None

        if result:
            st.markdown('<div class="slip-label">Wire Output</div>', unsafe_allow_html=True)
            for i, h in enumerate(result.candidates, start=1):
                st.markdown(
                    f"""<div class="ticket">
                        <div class="ticket-meta">CANDIDATE {i:02d} &middot; {h.char_count} CHARS &middot; FINE-TUNED</div>
                        <div class="ticket-head">{h.text}</div>""",
                    unsafe_allow_html=True,
                )

            if result.warnings:
                with st.expander("Desk notes (validation warnings)"):
                    render_warnings(result.warnings)
