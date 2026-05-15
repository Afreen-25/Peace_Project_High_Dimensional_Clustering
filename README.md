# 🎯 VidSim — YouTube Transcript Classifier via Embeddings + BoxSim

> **Convert YouTube video transcripts into high-dimensional text embeddings using the OpenAI API, then apply BoxSim locality-sensitive hashing to cluster and classify videos by semantic content — e.g. hate speech vs. peaceful discourse.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Pipeline Diagram](#pipeline-diagram)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Step-by-Step Usage](#step-by-step-usage)
  - [Step 1 — Fetch YouTube Transcripts](#step-1--fetch-youtube-transcripts)
  - [Step 2 — Chunk the Transcript](#step-2--chunk-the-transcript)
  - [Step 3 — Generate Text Embeddings (OpenAI)](#step-3--generate-text-embeddings-openai)
  - [Step 4 — Apply BoxSim Clustering](#step-4--apply-boxsim-clustering)
  - [Step 5 — Label Boxes (Hate / Peace / Neutral)](#step-5--label-boxes-hate--peace--neutral)
- [Embedding Models](#embedding-models)
- [BoxSim Algorithm](#boxsim-algorithm)
- [Classification Strategy](#classification-strategy)
- [Output Files](#output-files)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Limitations & Notes](#limitations--notes)
- [License](#license)

---

## Overview

This project builds a **zero-shot, unsupervised video classification pipeline** in four stages:

```
YouTube URL  →  Transcript  →  Text Embeddings  →  BoxSim Boxes  →  Label
```

Each video's transcript is split into ~2,000-character chunks. Every chunk is embedded into a high-dimensional vector (1,536 / 3,072 dims) using OpenAI's embedding API. BoxSim then binarises and clusters those vectors using locality-sensitive hashing. Because semantically similar chunks land in the same box, videos whose transcripts cluster together likely belong to the same content class.

**Example classes:**

| Label | Description |
|---|---|
| `HATE` | Dehumanising language, targeted hostility, incitement |
| `PEACE` | Constructive dialogue, reconciliation, de-escalation |
| `POLITICAL` | Partisan rhetoric, policy debate, civic argument |
| `NEUTRAL` | Factual reporting, educational content, commentary |

You supply seed examples for each class. BoxSim assigns every new video to the nearest box and inherits that box's majority label.

---

## How It Works

### 1 · Transcript Extraction
`youtube-transcript-api` fetches the auto-generated or human-edited captions for any public YouTube video. The raw caption segments are concatenated into a single plain-text transcript.

### 2 · Chunking
Long transcripts are split at sentence boundaries into ~2,000-character non-overlapping chunks. Each chunk becomes an independent unit of analysis — this preserves local semantic coherence while fitting within embedding model token limits.

### 3 · Embedding
Each chunk is passed to one of three OpenAI embedding models:

| Model | Dimensions | Use case |
|---|---|---|
| `text-embedding-3-small` | 1,536 | Fast, cost-efficient |
| `text-embedding-3-large` | 3,072 | Richest representation |
| `text-embedding-3-large` (truncated) | 768 | Balanced, curse-of-dimensionality mitigation |

Every raw embedding vector is then **min-max scaled** to `[−1, 1]` and **L2-normalised** to unit length, so cosine similarity becomes equivalent to dot product — exactly the metric BoxSim exploits.

### 4 · BoxSim
BoxSim applies a four-step transform to the embedding matrix:

1. **Binarise** — each dimension becomes `1` if positive, `0` otherwise.
2. **Rank bits** — sort dimensions by ascending variance; low-variance (stable) bits become the leading prefix bits.
3. **Sort rows** — lexicographic sort brings similar binary codes together.
4. **Form boxes** — group rows sharing the same first `k` bits into one box.

Chunks from different videos that land in the same box have similar semantic direction in the original embedding space.

### 5 · Classification
Each box is labelled by majority vote among any seed chunks it contains. Unlabelled boxes inherit the label of their nearest labelled box (Hamming distance on the `k`-bit prefix). Video-level labels are the majority vote across all their chunks' box labels.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT VIDEOS                           │
│           YouTube URL 1, URL 2, URL 3 …                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  youtube-transcript  │  ← youtube-transcript-api
              │      -api fetch      │
              └──────────┬───────────┘
                         │  raw caption text
                         ▼
              ┌──────────────────────┐
              │   Text Chunker       │  ← ~2 000 char / chunk
              │  (sentence-aware)    │     sentence boundary split
              └──────────┬───────────┘
                         │  N chunks per video
                         ▼
              ┌──────────────────────┐
              │  OpenAI Embeddings   │  ← text-embedding-3-small
              │  API  (per chunk)    │    text-embedding-3-large
              └──────────┬───────────┘    reduced (768-d)
                         │  float vectors
                         ▼
              ┌──────────────────────┐
              │  Scale + L2-Norm     │  ← min-max → [-1,1]
              │  (process_embedding) │    ÷ ‖v‖₂
              └──────────┬───────────┘
                         │  unit vectors
                         ▼
┌────────────────────────────────────────────────────────────┐
│                     BoxSim Pipeline                         │
│                                                            │
│   unit vectors                                             │
│       │                                                    │
│       ▼                                                    │
│   Binarise  →  B ∈ {0,1}^(N×D)                            │
│       │                                                    │
│       ▼                                                    │
│   Rank columns by variance  →  stable bits first           │
│       │                                                    │
│       ▼                                                    │
│   Sort rows lexicographically                              │
│       │                                                    │
│       ▼                                                    │
│   Form boxes  (prefix length k)                            │
│       │                                                    │
│       ▼                                                    │
│   Label boxes  (majority vote from seeds)                  │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Video-level label   │  ← majority vote over chunks
              │  HATE / PEACE /      │
              │  POLITICAL / NEUTRAL │
              └──────────────────────┘
```

---

## Repository Structure

```
vidsim/
│
├── README.md                        ← you are here
│
├── notebooks/
│   ├── text_embedding.ipynb         ← OpenAI embedding experiments
│   └── boxsim_demo.ipynb            ← BoxSim on toy 4-dim data
│
├── src/
│   ├── fetch_transcript.py          ← YouTube transcript fetcher
│   ├── chunker.py                   ← sentence-aware text chunker
│   ├── embed.py                     ← OpenAI embedding wrapper + normaliser
│   ├── boxsim.py                    ← BoxSim core (binarise / rank / box)
│   ├── classifier.py                ← seed-based box labelling + prediction
│   └── pipeline.py                  ← end-to-end runner
│
├── data/
│   ├── seeds/
│   │   ├── hate_seeds.txt           ← seed YouTube URLs for HATE class
│   │   └── peace_seeds.txt          ← seed YouTube URLs for PEACE class
│   └── embeddings_output.json       ← cached embeddings (auto-generated)
│
├── docs/
│   ├── BoxSim_Algorithm.pdf         ← algorithm + pseudocode reference
│   └── BoxSim_Algorithm.tex         ← LaTeX source
│
├── outputs/
│   └── boxsim_results_k*.png        ← box grid visualisations
│
├── requirements.txt
└── .env.example                     ← OPENAI_API_KEY template
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install

```bash
git clone https://github.com/your-username/vidsim.git
cd vidsim
pip install -r requirements.txt
```

**`requirements.txt`**
```
openai>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
youtube-transcript-api>=0.6.2
python-dotenv>=1.0.0
```

### API Key

```bash
cp .env.example .env
# then edit .env and paste your key:
# OPENAI_API_KEY=sk-...
```

Or export it directly:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

---

## Step-by-Step Usage

### Step 1 — Fetch YouTube Transcripts

```python
from src.fetch_transcript import get_transcript

url = "https://www.youtube.com/watch?v=VIDEO_ID"
transcript = get_transcript(url)
print(f"Transcript length: {len(transcript)} characters")
```

`get_transcript` uses `youtube-transcript-api` to pull captions. It prefers manually-created English subtitles and falls back to auto-generated ones.

```python
# src/fetch_transcript.py (core logic)
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(url: str) -> str:
    video_id = url.split("v=")[-1].split("&")[0]
    segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return " ".join(seg["text"] for seg in segments)
```

---

### Step 2 — Chunk the Transcript

```python
from src.chunker import chunk_text

chunks = chunk_text(transcript, chunk_size=2000)
print(f"Total chunks: {len(chunks)}")
```

The chunker splits at sentence boundaries so each chunk ends at a natural pause rather than mid-word:

```python
# src/chunker.py (core logic)
def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        boundary = max(text.rfind(". ", start, end), text.rfind(".\n", start, end))
        if boundary <= start:
            boundary = text.rfind(" ", start, end)
        if boundary <= start:
            boundary = end
        chunks.append(text[start:boundary + 1].strip())
        start = boundary + 1
    return [c for c in chunks if len(c) > 50]
```

---

### Step 3 — Generate Text Embeddings (OpenAI)

```python
from src.embed import embed_chunks

# Choose your model
results = embed_chunks(
    chunks=chunks,
    api_key="sk-...",          # or reads from OPENAI_API_KEY env var
    model="small",             # "small" | "large" | "reduced"
)

# results is a list of dicts:
# { "chunk_index": 0, "chunk_text": "...", "vector": np.array([...]) }
```

Internally, each raw embedding is processed by `process_embedding`:

```python
# src/embed.py (core logic)
import numpy as np

def process_embedding(raw: list[float]) -> np.ndarray:
    v = np.array(raw)
    # 1. Min-max scale to [-1, 1]
    v = 2 * (v - v.min()) / (v.max() - v.min()) - 1
    # 2. L2-normalise to unit length
    v = v / np.linalg.norm(v)
    return v
```

**Why this normalisation?**
- Scaling removes magnitude bias from OpenAI's raw outputs.
- L2-normalisation makes Hamming distance on the binary codes approximate cosine similarity — the natural metric for semantic similarity.

**Available models:**

```python
MODELS = {
    "small":   ("text-embedding-3-small", 1536),   # fastest, cheapest
    "large":   ("text-embedding-3-large", 3072),   # most expressive
    "reduced": ("text-embedding-3-large", 768),    # truncated to first 768 dims
}
```

**Embedding costs (approximate, May 2026):**

| Model | Dims | Cost per 1M tokens |
|---|---|---|
| `text-embedding-3-small` | 1,536 | ~$0.02 |
| `text-embedding-3-large` | 3,072 | ~$0.13 |

A typical 10-minute YouTube video produces ~15–20 chunks, so embedding a single video costs fractions of a cent.

---

### Step 4 — Apply BoxSim Clustering

```python
from src.boxsim import BoxSim

bsim = BoxSim(k=8)                   # k = prefix length
bsim.fit(vectors)                    # np.ndarray of shape (N, D)

labels = bsim.get_box_labels()       # dict: chunk_idx → box_prefix_string
summary = bsim.box_summary()         # DataFrame: box → member list + count
```

**Choosing `k`:**

| k | Boxes formed | Granularity |
|---|---|---|
| 1 | ≤ 2 | Very coarse — two halves of semantic space |
| 4 | ≤ 16 | Good starting point for small corpora |
| 8 | ≤ 256 | Balanced — recommended for 50–500 videos |
| 12 | ≤ 4,096 | Fine-grained — use for large corpora |
| 20 | ≤ 1M | Very fine — use with 10,000+ chunks |

**Visualise the box grid:**

```python
from src.boxsim import draw_box_grid

draw_box_grid(summary, k=8, model_name="small_1536")
```

Each cell in the grid shows:
```
Box: 10110011
# vecs: 7
Chunks: [2, 5, 9, 12, 14, 20, 23]
```

---

### Step 5 — Label Boxes (Hate / Peace / Neutral)

```python
from src.classifier import VideoClassifier

clf = VideoClassifier(boxsim=bsim)

# Register seed videos for each class
clf.add_seeds("hate",   hate_chunk_vectors,   hate_chunk_indices)
clf.add_seeds("peace",  peace_chunk_vectors,  peace_chunk_indices)
clf.add_seeds("neutral", neutral_vectors,     neutral_indices)

# Propagate labels to all boxes via Hamming-distance nearest-seed
clf.propagate_labels()

# Classify a new video
video_label = clf.classify_video(new_video_chunk_vectors)
# → "hate" | "peace" | "neutral" | "political"
```

**How label propagation works:**

1. Seed chunks land in boxes — those boxes inherit the seed's class label.
2. Unlabelled boxes find their nearest labelled box by Hamming distance on the `k`-bit prefix.
3. Each video's final label is the majority vote across all its chunks' box labels.

---

## Embedding Models

The notebook (`text_embedding.ipynb`) experiments with three configurations:

### `text-embedding-3-small` — 1,536 dims
```python
client.embeddings.create(model="text-embedding-3-small", input=chunk)
```
- Best for: exploratory clustering, rapid iteration, cost-sensitive pipelines.
- Characteristic: values tend to be larger in magnitude per dimension because signal is concentrated into fewer slots. Early dimensions are strongly activated by topic signals (e.g., formal political rhetoric → high first-5 values).

### `text-embedding-3-large` — 3,072 dims
```python
client.embeddings.create(model="text-embedding-3-large", input=chunk)
```
- Best for: fine-grained distinction between similar content types (e.g., heated political debate vs. actual incitement).
- Characteristic: smaller per-dimension magnitudes — semantic weight is diffused across 3,072 slots, capturing subtler nuance. Higher API cost.

### Reduced — 768 dims (truncation of large)
```python
raw_large = client.embeddings.create(model="text-embedding-3-large", input=chunk)
reduced = np.array(raw_large[:768])
```
- Best for: mitigating the curse of dimensionality in BoxSim. 768 dims still captures strong signal while keeping bit matrices tractable.
- Note: this is a heuristic truncation, not a trained dimensionality reduction. PCA or UMAP would be more principled alternatives.

---

## BoxSim Algorithm

See [`docs/BoxSim_Algorithm.pdf`](docs/BoxSim_Algorithm.pdf) for the full formal write-up.

**Quick reference:**

```
Input:  Matrix X ∈ ℝ^(N×D)  — N chunks, D embedding dims

1. L2-normalise each row:        x̂ᵢ = xᵢ / ‖xᵢ‖₂
2. Binarise:                     Bᵢⱼ = 1 if x̂ᵢⱼ > 0 else 0
3. Rank columns by variance:     π = argsort(σ²ⱼ, ascending)
4. Reorder columns:              B′ = B[:, π]
5. Sort rows lexicographically:  B̂ = lexsort(B′)
6. Form boxes (prefix k):        box(i) = join(B̂ᵢ₀ … B̂ᵢ,ₖ₋₁)

Query: apply steps 1–3 to query vector q, extract k-bit prefix, return box.
Complexity: O(ND + N log N) preprocessing,  O(1) + O(|box|) query.
```

**Why low-variance bits come first:**
A bit column with near-zero variance is almost constant across all chunks — it partitions the space very cleanly into two groups. By leading with these stable bits, the first `k`-bit prefix captures the most reliable structural divisions of the semantic space before adding noisier, higher-variance bits.

---

## Classification Strategy

### Approach A — Seed-based (recommended for small labelled sets)

Manually label 5–10 representative videos per class. Their chunks become seeds. All other boxes are labelled by Hamming-nearest seed box.

### Approach B — Keyword anchor

Provide a short keyword list per class (e.g., `["kill", "destroy", "inferior"]` for hate). Embed the keyword string, find its box, and propagate that label. No labelled videos required.

### Approach C — Threshold on box purity

For each box, compute the fraction of chunks containing class-specific keywords. If `> threshold` (e.g. 0.6), assign that class. Ambiguous boxes are labelled `MIXED`.

### Reading the results

```
Video: "Senator floor speech on immigration"
  Chunk 0  → Box 10110011  → label: POLITICAL
  Chunk 1  → Box 10110011  → label: POLITICAL
  Chunk 2  → Box 11001100  → label: POLITICAL
  Chunk 3  → Box 01010101  → label: NEUTRAL
  ─────────────────────────────────────────
  Video label: POLITICAL  (3/4 chunks)
```

---

## Output Files

| File | Description |
|---|---|
| `data/embeddings_output.json` | All chunk embeddings for all videos and all three models |
| `outputs/boxsim_<model>_k<k>.png` | Box grid visualisation for each model × k combination |
| `outputs/classifications.csv` | Final per-video labels with confidence (chunk vote fraction) |

**`embeddings_output.json` schema:**
```json
[
  {
    "chunk_index": 0,
    "chunk_text": "Madam President, we often consider ...",
    "chunk_length": 1987,
    "small_1536":   [0.068, 0.045, -0.007, ...],
    "large_3072":   [0.039, -0.020, -0.019, ...],
    "reduced_768":  [0.061, -0.032, -0.030, ...]
  },
  ...
]
```

---

## Configuration Reference

```python
# pipeline.py — top-level configuration
CONFIG = {
    # Embedding
    "model":       "small",     # "small" | "large" | "reduced"
    "chunk_size":  2000,        # characters per chunk

    # BoxSim
    "k":           8,           # prefix length (bits)
    "top_bits":    20,          # number of top bits to use in the DataFrame

    # Classification
    "strategy":    "seed",      # "seed" | "keyword" | "threshold"
    "threshold":   0.6,         # for strategy="threshold"
    "min_chunks":  3,           # minimum chunks for a reliable video label

    # Output
    "save_grid":   True,        # save box grid PNGs
    "save_json":   True,        # save embeddings JSON
}
```

---

## Examples

### Classify a single video

```bash
python src/pipeline.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --model small \
  --k 8
```

### Classify a batch of videos from a file

```bash
# urls.txt — one URL per line
python src/pipeline.py \
  --batch urls.txt \
  --model large \
  --k 12 \
  --output outputs/results.csv
```

### Interactive notebook walkthrough

```bash
jupyter notebook notebooks/text_embedding.ipynb
```

The notebook walks through:
1. Embedding a single transcript with all three models.
2. Comparing first-5 dimension values across models.
3. Running BoxSim for `k = 1, 2, 4, 8` and visualising the grids.

---

## Limitations & Notes

- **Binary quantisation loses magnitude.** BoxSim treats direction only. Two chunks with identical positive/negative sign patterns land in the same box even if their magnitudes differ greatly. For magnitude-sensitive tasks, consider cosine similarity search (FAISS, Annoy) as a post-filter.

- **Transcript quality varies.** Auto-generated captions can contain errors, especially for accented speech or technical vocabulary. Pre-processing with a spell-corrector or Whisper transcription improves embedding quality.

- **k is a hyperparameter.** There is no universally correct k. Start at `k=4` and increase until boxes become semantically coherent on your validation set. Visualise the grids.

- **OpenAI truncation limit.** `text-embedding-3-small` has a maximum input of 8,191 tokens. A 2,000-character chunk is typically ~400–600 tokens, well within limits.

- **The reduced 768-dim model is a heuristic.** Truncating the first 768 dimensions of the large model's output is a practical shortcut, not a trained projection. If dimensionality reduction is critical, use PCA fitted on your corpus.

- **Classification is unsupervised by default.** Without seeds, BoxSim only groups — it does not label. Labels require human annotation of at least a few representative examples per class.

---

## License

MIT License — see `LICENSE` for details.

---

*Built on top of [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) and the BoxSim locality-sensitive hashing algorithm. See [`docs/BoxSim_Algorithm.pdf`](docs/BoxSim_Algorithm.pdf) for the full algorithm specification and pseudocode.*
