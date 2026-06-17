import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from wordcloud import WordCloud

from src.sentiment import score_to_float


def plot_sentiment_progression(results: list[dict], title: str = "Sentiment Progression"):
    scores = [score_to_float(r) for r in results]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(scores, linewidth=1.4, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s > 0 for s in scores], alpha=0.25, color="green", label="Positive")
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s <= 0 for s in scores], alpha=0.25, color="red", label="Negative")
    ax.set_xlabel("Text chunk")
    ax.set_ylabel("Sentiment score")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_section_sentiment(section_avgs: dict[str, float], title: str = "Section Sentiment"):
    sections = list(section_avgs.keys())
    scores = list(section_avgs.values())
    colors = ["green" if s > 0 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sections) * 0.55)))
    y = range(len(sections))
    ax.barh(list(y), scores, color=colors, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(sections, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average sentiment score")
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    return fig


def plot_article_comparison(article_scores: dict[str, float]):
    titles = list(article_scores.keys())
    scores = list(article_scores.values())
    colors = ["green" if s > 0 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(titles, scores, color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average sentiment score")
    ax.set_title("Article Sentiment Comparison")
    ax.set_ylim(-1, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_knowledge_graph(
    nodes: list[str],
    edges: list[tuple[str, str]],
    scores: dict[str, float],
    title: str = "Sentiment Knowledge Graph",
):
    G = nx.DiGraph()
    node_set = set(nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from([(s, d) for s, d in edges if s in node_set and d in node_set])

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    node_colors = [cmap(norm(scores.get(n, 0.0))) for n in G.nodes()]
    node_sizes = [400 + 300 * abs(scores.get(n, 0.0)) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, seed=42, k=1.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, arrows=True,
                           arrowsize=8, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Sentiment score", shrink=0.5)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_wordcloud(text: str, title: str = "Word Cloud"):
    wc = WordCloud(width=900, height=450, background_color="white",
                   max_words=200, colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig
