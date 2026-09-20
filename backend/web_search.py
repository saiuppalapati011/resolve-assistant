"""Small, optional web-search fallback for documentation questions.

This intentionally uses DuckDuckGo's HTML endpoint so the college-project
version does not need another API key. Results are treated as untrusted
context and the final answer is still written by the selected LLM with links
included for the user to verify.
"""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urlparse

import requests

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = (attrs_dict.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": "", "snippet": ""}
            self._current["url"] = _unwrap_result_url(attrs_dict.get("href", ""))
            self._in_title = True
        elif self._current is not None and "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_title = False
        if self._current is not None and self._in_snippet and tag in {"a", "div", "td"}:
            self._in_snippet = False
            if self._current["url"] and self._current["title"]:
                self.results.append(self._current)
                self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet:
            self._current["snippet"] += data


def _unwrap_result_url(value: str | None) -> str:
    value = value or ""
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        return parse_qs(parsed.query).get("uddg", [value])[0]
    return value


def search_web(query: str, limit: int | None = None) -> list[dict[str, str]]:
    """Return a few web results, preferring official Blackmagic pages."""
    if not settings.web_search_enabled:
        return []

    max_results = limit or settings.web_search_max_results
    queries = [
        f"site:blackmagicdesign.com DaVinci Resolve {query}",
        f"DaVinci Resolve {query}",
    ]
    collected: list[dict[str, str]] = []
    seen: set[str] = set()

    for search_query in queries:
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/?q=" + quote_plus(search_query),
                headers={"User-Agent": "ResolveAssistant/1.0"},
                timeout=settings.web_search_timeout,
            )
            response.raise_for_status()
            parser = _DuckDuckGoParser()
            parser.feed(response.text)
        except requests.RequestException as exc:
            logger.warning("Web search request failed", error=str(exc))
            continue

        for result in parser.results:
            url = result.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            collected.append(result)
            if len(collected) >= max_results:
                return collected

    return collected
