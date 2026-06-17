"""
Knowledge-graph analytics built on top of networkx.

Provides:
  - build_digraph    : convert (nodes, edges) from wiki_loader into a DiGraph
  - compute_centrality : degree, in-degree, betweenness, PageRank
  - centrality_sentiment_table : DataFrame correlating centrality with sentiment
"""

from __future__ import annotations

import pandas as pd
import networkx as nx


def build_digraph(nodes: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    node_set = set(nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from((s, d) for s, d in edges if s in node_set and d in node_set)
    return G


def compute_centrality(G: nx.DiGraph) -> dict[str, dict[str, float]]:
    return {
        "degree": nx.degree_centrality(G),
        "in_degree": nx.in_degree_centrality(G),
        "out_degree": nx.out_degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "pagerank": nx.pagerank(G, alpha=0.85),
    }


def centrality_sentiment_table(
    G: nx.DiGraph,
    scores: dict[str, float],
) -> pd.DataFrame:
    """Build a DataFrame with one row per node, columns for every centrality metric
    and the sentiment score. Sorted by PageRank descending."""
    centrality = compute_centrality(G)
    rows = []
    for node in G.nodes():
        row: dict[str, float | str] = {"article": node, "sentiment": scores.get(node, 0.0)}
        for metric, values in centrality.items():
            row[metric] = round(values.get(node, 0.0), 5)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("pagerank", ascending=False).reset_index(drop=True)


def high_sentiment_hubs(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    centrality_col: str = "pagerank",
    top_n: int = 5,
) -> pd.DataFrame:
    """Articles that combine high centrality with extreme (positive or negative) sentiment."""
    df = df.copy()
    df["abs_sentiment"] = df[sentiment_col].abs()
    df["hub_score"] = df[centrality_col] * df["abs_sentiment"]
    return df.nlargest(top_n, "hub_score")[
        ["article", sentiment_col, centrality_col, "hub_score"]
    ]
