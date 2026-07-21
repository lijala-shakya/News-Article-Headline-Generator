"""Fetches a URL and extracts the main article text, so it can be fed into
the same pipeline as pasted text (see pipeline.generate_headlines /
generate_headlines_local).

Kept intentionally simple: one function in, one string out. All downstream
code (schemas.ArticleInput, validators, both model backends) is unchanged --
scraping is just a new way to produce `article_text`, not a new code path
through the pipeline.
"""

import urllib.robotparser
from urllib.parse import urlparse

import requests
import trafilatura

USER_AGENT = "HeadlineDeskBot/1.0 (+https://example.com/bot-info)"
REQUEST_TIMEOUT = 10
MIN_EXTRACTED_CHARS = 20


def _robots_allowed(url: str) -> bool:
    """Best-effort robots.txt check. If robots.txt can't be fetched or
    parsed, fail open (allow) rather than blocking legitimate use -- this
    mirrors how most browsers/readers behave, but errs toward caution by
    still checking when the file is reachable."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        return True
    return rp.can_fetch(USER_AGENT, url)


def scrape_article(url: str) -> str:
    """Fetches `url` and returns the extracted main article text.

    Raises RuntimeError with a user-facing message on any failure --
    unreachable URL, disallowed by robots.txt, or no extractable article
    content (e.g. paywalled, JS-only, or not an article page at all).
    """
    if not url.strip().lower().startswith(("http://", "https://")):
        raise RuntimeError("Please provide a full URL starting with http:// or https://")

    if not _robots_allowed(url):
        raise RuntimeError(
            "This site's robots.txt disallows automated fetching of that page."
        )

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not fetch the URL: {exc}") from exc

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not extracted or len(extracted.strip()) < MIN_EXTRACTED_CHARS:
        raise RuntimeError(
            "Couldn't extract readable article text from that page -- it "
            "may be paywalled, JS-rendered, or not an article page."
        )

    return extracted.strip()
