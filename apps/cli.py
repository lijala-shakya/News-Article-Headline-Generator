"""Command-line entry point — run with: python -m src.cli"""

import sys

from apps.pipeline import generate_headlines
from apps.scraper import scrape_article


def main() -> None:
    print("=== News Headline Generator (CLI) ===")
    source = input("Source: paste text or fetch from a URL? [text/url] (default: text): ").strip().lower() or "text"

    if source == "url":
        url = input("Article URL: ").strip()
        print("Fetching and extracting article text...")
        try:
            article_text = scrape_article(url)
        except RuntimeError as e:
            print(f"\nError: {e}")
            sys.exit(1)
        print(f"\nExtracted {len(article_text)} characters.")
    else:
        print("Paste your article text below, then press Enter twice to submit:\n")

        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        article_text = "\n".join(lines).strip()

    if not article_text:
        print("No article text provided. Exiting.")
        sys.exit(1)

    style = input("Style [neutral/clickbait/seo] (default: neutral): ").strip() or "neutral"

    try:
        result = generate_headlines(article_text=article_text, style=style)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print(f"\n--- Headlines ({result.style.value}) ---")
    for i, h in enumerate(result.candidates, start=1):
        print(f"{i}. {h.text}  [{h.char_count} chars]")

    if result.warnings:
        print("\n--- Warnings ---")
        for w in result.warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()