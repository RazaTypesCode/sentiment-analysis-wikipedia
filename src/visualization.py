"""
All visualization functions for Wikipedia Language Analytics.

Each function returns a matplotlib Figure so the caller can either call
plt.show() interactively or fig.savefig(...) to save.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import seaborn as sns
from wordcloud import WordCloud

from src.sentiment import score_to_float


# ── Sentiment progression ──────────────────────────────────────────────────────

def plot_sentiment_progression(
    results: list[dict],
    title: str = "Sentiment Progression",
) -> plt.Figure:
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


# ── Section sentiment ──────────────────────────────────────────────────────────

def plot_section_sentiment(
    section_avgs: dict[str, float],
    title: str = "Section Sentiment",
) -> plt.Figure:
    sections = list(section_avgs.keys())
    scores = list(section_avgs.values())
    colors = ["green" if s > 0 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sections) * 0.55)))
    y = list(range(len(sections)))
    ax.barh(y, scores, color=colors, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(sections, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average sentiment score")
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    return fig


# ── Article comparison ─────────────────────────────────────────────────────────

def plot_article_comparison(article_scores: dict[str, float]) -> plt.Figure:
    titles = list(article_scores.keys())
    scores = list(article_scores.values())
    colors = ["green" if s > 0 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(max(8, len(titles) * 1.3), 5))
    ax.bar(titles, scores, color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average sentiment score")
    ax.set_title("Article Sentiment Comparison")
    ax.set_ylim(-1, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


# ── Group box plot ─────────────────────────────────────────────────────────────

def plot_group_boxplot(
    groups: dict[str, list[float]],
    title: str = "Group Sentiment Comparison",
) -> plt.Figure:
    """Box + strip plot comparing sentiment distributions across named groups."""
    import pandas as pd
    rows = [{"group": name, "sentiment": s}
            for name, scores in groups.items() for s in scores]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 2), 5))
    sns.boxplot(data=df, x="group", y="sentiment", ax=ax,
                palette="Set2", width=0.5)
    sns.stripplot(data=df, x="group", y="sentiment", ax=ax,
                  color="black", size=5, alpha=0.6, jitter=True)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Sentiment score")
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


# ── Entity sentiment ───────────────────────────────────────────────────────────

def plot_entity_sentiment(
    entity_scores: dict[str, float],
    title: str = "Entity Sentiment",
) -> plt.Figure:
    entities = list(entity_scores.keys())
    scores = list(entity_scores.values())
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    entities = [entities[i] for i in order]
    scores = [scores[i] for i in order]
    colors = ["green" if s > 0 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, len(entities) * 0.5)))
    y = list(range(len(entities)))
    ax.barh(y, scores, color=colors, alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(entities, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sentiment score around entity mentions")
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    return fig


# ── Emotion bars ───────────────────────────────────────────────────────────────

_EMOTION_COLORS = {
    "anger":   "#d62728",
    "disgust": "#8c564b",
    "fear":    "#9467bd",
    "sadness": "#1f77b4",
    "neutral": "#7f7f7f",
    "surprise":"#ff7f0e",
    "joy":     "#2ca02c",
}


def plot_emotion_bars(
    emotion_scores: dict[str, float],
    title: str = "Emotion Profile",
) -> plt.Figure:
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    colors = [_EMOTION_COLORS.get(l, "steelblue") for l in labels]
    order = sorted(range(len(values)), key=lambda i: -values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Average score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def plot_emotion_heatmap(
    profiles: dict[str, dict[str, float]],
    title: str = "Emotion Heatmap",
) -> plt.Figure:
    """Heatmap of emotion scores across multiple articles."""
    import pandas as pd
    df = pd.DataFrame(profiles).T.fillna(0)
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.2), max(4, len(df) * 0.5)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── Radar chart ────────────────────────────────────────────────────────────────

def plot_radar(
    groups: dict[str, dict[str, float]],
    title: str = "Multi-Metric Radar",
) -> plt.Figure:
    """Overlay radar charts for one or more groups.

    groups: {'Group A': {'metric1': val, ...}, 'Group B': {...}}
    All groups must have the same metric keys.
    Values are min-max normalised per metric before plotting.
    """
    all_keys = list(next(iter(groups.values())).keys())
    n = len(all_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise per metric across all groups
    raw = {m: [g[m] for g in groups.values()] for m in all_keys}
    mn = {m: min(v) for m, v in raw.items()}
    mx = {m: max(v) for m, v in raw.items()}

    def norm(m, v):
        span = mx[m] - mn[m]
        return (v - mn[m]) / span if span else 0.5

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    palette = plt.cm.tab10.colors

    for idx, (name, metrics) in enumerate(groups.items()):
        values = [norm(m, metrics[m]) for m in all_keys] + [norm(all_keys[0], metrics[all_keys[0]])]
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_keys, size=9)
    ax.set_yticklabels([])
    ax.set_title(title, size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig


# ── Knowledge graph ────────────────────────────────────────────────────────────

def plot_knowledge_graph(
    nodes: list[str],
    edges: list[tuple[str, str]],
    scores: dict[str, float],
    title: str = "Sentiment Knowledge Graph",
) -> plt.Figure:
    G = nx.DiGraph()
    node_set = set(nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from((s, d) for s, d in edges if s in node_set and d in node_set)

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


# ── Centrality vs sentiment scatter ───────────────────────────────────────────

def plot_centrality_scatter(
    df,
    centrality_col: str = "pagerank",
    sentiment_col: str = "sentiment",
    title: str = "Centrality vs Sentiment",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["green" if s > 0 else "red" for s in df[sentiment_col]]
    ax.scatter(df[centrality_col], df[sentiment_col], c=colors, alpha=0.7, s=60)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(centrality_col.replace("_", " ").title())
    ax.set_ylabel("Sentiment score")
    ax.set_title(title)

    for _, row in df.iterrows():
        ax.annotate(
            row["article"][:20],
            (row[centrality_col], row[sentiment_col]),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )
    plt.tight_layout()
    return fig


# ── Temporal sentiment ─────────────────────────────────────────────────────────

def plot_temporal_sentiment(
    timestamps: list[str],
    scores: list[float],
    title: str = "Sentiment Over Time",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(len(scores)), scores, marker="o", linewidth=1.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s > 0 for s in scores], alpha=0.2, color="green")
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s <= 0 for s in scores], alpha=0.2, color="red")
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels([t[:10] for t in timestamps], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Average sentiment score")
    ax.set_ylim(-1, 1)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── Word cloud ─────────────────────────────────────────────────────────────────

def plot_wordcloud(text: str, title: str = "Word Cloud") -> plt.Figure:
    wc = WordCloud(width=900, height=450, background_color="white",
                   max_words=200, colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig
