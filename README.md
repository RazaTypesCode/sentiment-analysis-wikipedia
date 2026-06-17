# Wikipedia Language Analytics

> Quantitative Analysis of Language and Sentiment Patterns in Wikipedia Articles

Multi-dimensional NLP + graph analytics project. Articles are fetched live from the Wikipedia API — no dataset required.

---

## What it does

| Section | Analysis |
|---|---|
| 1 | Sentiment progression across a single article (RoBERTa) |
| 2 | 7-class emotion profile (anger, fear, joy, sadness, etc.) |
| 3 | Section-wise sentiment comparison |
| 4 | Language metrics: readability, lexical diversity, term densities |
| 5 | Entity-level sentiment via spaCy NER |
| 6 | Comparative group analysis with radar charts and emotion heatmaps |
| 7 | Statistical testing — Welch t-test + Mann-Whitney U |
| 8 | Sentiment knowledge graph with PageRank centrality |
| 9 | Temporal analysis via Wikipedia revision history |
| 10 | Word cloud |

---

## Repository structure

```
wikipedia-sentiment-analysis/
├── notebooks/
│   └── sentiment_analysis.ipynb   ← main Colab notebook
├── src/
│   ├── article_sets.py            ← curated article collections by category
│   ├── wiki_loader.py             ← Wikipedia API + revision history + citation count
│   ├── sentiment.py               ← RoBERTa sentiment + DistilRoBERTa emotion
│   ├── text_metrics.py            ← readability, lexical diversity, term densities
│   ├── entity_sentiment.py        ← spaCy NER + context-window entity scoring
│   ├── graph_analytics.py         ← networkx centrality + sentiment correlation table
│   ├── statistics.py              ← Welch t-test, Mann-Whitney U, pairwise comparisons
│   ├── visualization.py           ← all plot functions
│   └── utils.py                   ← text chunking helpers
├── requirements.txt
└── .gitignore
```

---

## Quick start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Paste and run:

```python
!git clone https://github.com/RazaTypesCode/sentiment-analysis-wikipedia.git
%cd sentiment-analysis-wikipedia
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm -q
```

3. Open and run `notebooks/sentiment_analysis.ipynb`.

**Recommended:** GPU runtime (Runtime → Change runtime type → T4 GPU). The RoBERTa sentiment model is ~1.3 GB and runs significantly faster on GPU.

---

## Models

| Task | Model | Why |
|---|---|---|
| Sentiment | `siebert/sentiment-roberta-large-english` | Trained on diverse sources: news, reviews, social media — better fit for Wikipedia than SST-2 |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 7 labels: anger, disgust, fear, joy, neutral, sadness, surprise |

Text is chunked into 400-character windows (safely within the 512-token limit). Pipeline uses `batch_size=16` for efficient GPU throughput.

Sentiment scores are mapped to `[-1, +1]`:
```
POSITIVE, confidence 0.92  →  +0.92
NEGATIVE, confidence 0.85  →  −0.85
```

---

## Article groups

`src/article_sets.py` contains curated comparison sets:

| Set | Group A | Group B |
|---|---|---|
| Civil Wars | Global South (8 articles) | Global North (8 articles) |
| Independence movements | Africa | Europe |
| Political leaders | Africa | Europe |
| Scientists | Western | Non-Western |
| Historical empires | European | Asian |
| Topic themes | Technology | Environment |

---

## Statistical approach

```
H₀: mean sentiment scores for group A = group B
H₁: means differ (two-sided)
```

- **Welch t-test** — does not assume equal variance
- **Mann-Whitney U** — non-parametric, robust to small or non-normal samples

Results flagged significant at `p < 0.05` (★) and `p < 0.01` (★★).

---

## Knowledge graph

BFS from a seed article along Wikipedia hyperlinks. Default:

```python
SEED      = 'Quantum mechanics'
DEPTH     = 1    # hops from seed
MAX_LINKS = 8    # links followed per article
```

Each node is scored for sentiment. The graph is then analyzed with:
- PageRank, betweenness, degree centrality
- A hub score (`pagerank × |sentiment|`) identifies influential pages with strong sentiment

---

## Temporal analysis

The MediaWiki API exposes full revision history. For each revision, wikitext is fetched and stripped with `mwparserfromhell`, then scored. The result is a time-series plot of how sentiment has changed across edits.

---

## Requirements

```
wikipedia-api    transformers     torch           pandas
numpy            matplotlib       seaborn         networkx
wordcloud        spacy            scipy           textstat
requests         mwparserfromhell
```
