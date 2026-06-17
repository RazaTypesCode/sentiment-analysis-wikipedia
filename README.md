# Wikipedia Sentiment Analysis

Sentiment analysis and visualization on Wikipedia articles, including a knowledge-graph view where sentiment propagates across linked pages.

No dataset required — articles are fetched live from the Wikipedia API.

---

## What it does

| Analysis | Description |
|---|---|
| Sentiment progression | How sentiment shifts chunk-by-chunk through a single article |
| Section-wise sentiment | Which sections of an article are most positive/negative |
| Article comparison | Side-by-side average sentiment for a set of articles |
| Sentiment knowledge graph | Crawl Wikipedia links, score every reachable article, visualize as a colored network |
| Word cloud | Dominant vocabulary in an article |

---

## Repository structure

```
wikipedia-sentiment-analysis/
├── notebooks/
│   └── sentiment_analysis.ipynb   ← main Colab notebook
├── src/
│   ├── wiki_loader.py             ← Wikipedia API helpers + graph BFS crawler
│   ├── sentiment.py               ← DistilBERT sentiment pipeline
│   ├── visualization.py           ← all plot functions
│   └── utils.py                   ← text chunking helpers
├── requirements.txt
└── .gitignore
```

---

## Quick start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Paste and run the setup cell:

```python
!git clone https://github.com/RazaTypesCode/sentiment-analysis-wikipedia.git
%cd sentiment-analysis-wikipedia
!pip install -q -r requirements.txt
```

3. Open `notebooks/sentiment_analysis.ipynb` — or just run the cells in order from your new notebook after the setup above.

**Recommended:** use a GPU runtime (Runtime → Change runtime type → T4 GPU) to speed up the transformer inference.

---

## Model

[DistilBERT fine-tuned on SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) via Hugging Face `transformers`. No training required.

Text is chunked into 400-character windows (well within the 512-token limit) and each chunk is classified independently. The score for each chunk is mapped to `[-1, 1]`:

```
POSITIVE, confidence 0.92  →  +0.92
NEGATIVE, confidence 0.85  →  -0.85
```

---

## Knowledge graph

The graph crawler (`src/wiki_loader.build_graph_data`) does a BFS from a seed article, following Wikipedia hyperlinks up to a configurable depth. Default settings:

```python
SEED      = 'Quantum mechanics'
DEPTH     = 1    # hops from seed
MAX_LINKS = 8    # links followed per article
```

This gives ~9 nodes and ~8 edges — fast enough to run without a GPU. Increasing depth to 2 with max_links=10 yields ~111 nodes and takes several minutes on a GPU.

---

## Requirements

```
wikipedia-api
transformers
torch
pandas
numpy
matplotlib
seaborn
networkx
wordcloud
```
