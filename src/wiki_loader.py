"""
Wikipedia data fetching.

Three layers:
  1. wikipedia-api  — modern article text and sections
  2. Wikimedia REST API (requests) — revision history
  3. MediaWiki Action API (requests) — raw wikitext for citation counting
"""

from __future__ import annotations

import re

import requests
import wikipediaapi
import mwparserfromhell

_wiki = wikipediaapi.Wikipedia(
    user_agent="wikipedia-sentiment-analysis/1.0",
    language="en",
)

_HEADERS = {"User-Agent": "wikipedia-sentiment-analysis/1.0"}
_API = "https://en.wikipedia.org/w/api.php"


# ── Article text ───────────────────────────────────────────────────────────────

def get_article(title: str) -> str:
    page = _wiki.page(title)
    if not page.exists():
        raise ValueError(f"Article '{title}' not found on Wikipedia")
    return page.text


def get_sections(title: str) -> dict[str, str]:
    page = _wiki.page(title)
    if not page.exists():
        raise ValueError(f"Article '{title}' not found on Wikipedia")
    return {s.title: s.text for s in page.sections if s.text.strip()}


# ── Graph crawling ─────────────────────────────────────────────────────────────

def get_links(title: str, max_links: int = 15) -> list[str]:
    page = _wiki.page(title)
    if not page.exists():
        return []
    return list(page.links.keys())[:max_links]


def build_graph_data(
    seed_title: str,
    depth: int = 1,
    max_links: int = 10,
) -> tuple[list[str], list[tuple[str, str]]]:
    """BFS from seed_title up to `depth` hops.

    Returns (nodes, edges) where edges are (src, dst) pairs.
    """
    visited: set[str] = set()
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    queue: list[tuple[str, int]] = [(seed_title, 0)]

    while queue:
        title, level = queue.pop(0)
        if title in visited:
            continue
        visited.add(title)
        nodes.append(title)

        if level < depth:
            for link in get_links(title, max_links):
                edges.append((title, link))
                if link not in visited:
                    queue.append((link, level + 1))

    return nodes, edges


# ── Citation count ─────────────────────────────────────────────────────────────

def count_references(title: str) -> int:
    """Return the number of <ref> tags in the article wikitext."""
    resp = requests.get(
        _API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
        },
        headers=_HEADERS,
        timeout=15,
    )
    wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
    return len(re.findall(r"<ref", wikitext, re.IGNORECASE))


# ── Revision history ───────────────────────────────────────────────────────────

def get_revision_ids(title: str, limit: int = 10) -> list[dict]:
    """Return up to `limit` revisions for an article, newest first.

    Each entry: {'revid': int, 'timestamp': str}
    """
    resp = requests.get(
        _API,
        params={
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "ids|timestamp",
            "rvlimit": limit,
            "format": "json",
        },
        headers=_HEADERS,
        timeout=15,
    )
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("revisions", [])


def get_article_by_revision(revid: int) -> str:
    """Fetch and return plain text for a specific Wikipedia revision."""
    resp = requests.get(
        _API,
        params={
            "action": "query",
            "prop": "revisions",
            "revids": revid,
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        },
        headers=_HEADERS,
        timeout=30,
    )
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    wikitext = page["revisions"][0]["slots"]["main"]["*"]
    return mwparserfromhell.parse(wikitext).strip_code()
