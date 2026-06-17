import wikipediaapi

_wiki = wikipediaapi.Wikipedia(
    user_agent='wikipedia-sentiment-analysis/1.0',
    language='en'
)


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
    Only follows links to articles that actually exist.
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
