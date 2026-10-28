# Mental Health Analysis using Social Media Data
## Contextual Emotion Distribution Vectorizer (CEDV)
### Architectural & Implementation Plan (Streamlined 8-Class Architecture)

---

## 1. Project Overview & 8-Class Taxonomy

Traditional text classification treats mental health detection as a rigid, mutually exclusive single-label task. In real-world social discourse, emotional expression is continuous and multi-faceted. The **Contextual Emotion Distribution Vectorizer (CEDV)** outputs an 8-dimensional normalized distribution vector modeling everyday emotional and psychological states:

| Class Index | Target Label | Source Subreddits | Psychological Focus |
| :---: | :--- | :--- | :--- |
| `0` | `depression` | `r/depression` | Persistent low mood, lethargy, loss of purpose, emotional numbness |
| `1` | `anxiety` | `r/anxiety`, `r/socialanxiety` | Panic, generalized worry, social apprehension, physiological tension |
| `2` | `stress_burnout` | `r/stress`, `r/burnout`, `r/offmychest` | Overwork exhaustion, emotional exhaustion, feeling overwhelmed |
| `3` | `adhd` | `r/adhd` | Executive dysfunction, focus/dopamine struggles, hyperactivity |
| `4` | `autism` | `r/autism` | Neurodivergence, sensory overload, masking, social communication |
| `5` | `loneliness` | `r/lonely` | Isolation, lack of belonging, alienation |
| `6` | `anger` | `r/anger`, `r/vent` | Irritation, indignation, venting, emotional reactivity |
| `7` | `positive_neutral` | `r/happy`, `r/casualconversation` | Uplifting, grounded, gratitude, everyday non-clinical discourse |

---

## 2. Directory Structure

```
Mental-Health-Analysis-using-Social-Media-Data/
├── README.md                   # Project documentation & architectural specification
├── PLAN.md                     # Synchronized architectural & execution roadmap
├── requirements.txt            # Minimal dependencies (tensorflow, pandas, requests, rich, scikit-learn)
├── .gitignore                  # Clean Python, TensorFlow, and cache ignore rules
├── data/
│   └── dataset.csv             # Unified training dataset (scraped + cached)
├── src/
│   ├── scraper.py              # Keyless scraper (Arctic Shift -> Reddit .json/.rss -> Seed fallback)
│   ├── preprocess.py           # Text cleaning, label encoding, tf.data dataset generator
│   ├── model.py                # Native Keras TextVectorization + BiLSTM + 1D-CNN + Distribution Head
│   └── train.py                # Mixed precision (FP16), AdamW, training loop, evaluation metrics
└── cli.py                      # Interactive CLI: direct text analysis and live subreddit fetch
```

---

## 3. Keyless Data Scraping Strategy (Recent Posts 2024–2026)

The acquisition engine operates under strict zero-authentication constraints (zero API keys, zero OAuth tokens, zero paid tiers):

- **Target Time Horizon:** Recent posts (2024–2026) to reflect modern slang, vernacular, and contemporary context.
- **Cascading Zero-Auth Data Acquisition:**
  1. **Primary (Arctic Shift Search API):** Queries `https://arctic-shift.photon-reddit.com/api/posts/search?subreddit={sub}&limit={n}&after=1704067200` to fetch unauthenticated submissions with titles, selftext, scores, and timestamps.
  2. **Secondary (Public Reddit .json / .rss):** Unauthenticated direct requests to `https://www.reddit.com/r/{sub}/new.json` or `.rss` with custom research user agent and backoff delay.
  3. **Embedded Fallback:** Built-in seed samples in `src/scraper.py` mapping to all 8 classes ensuring offline and rate-limit safety.
- **Unified CSV Cache:** Scraped rows are automatically deduplicated via MD5 content hashing and written to `data/dataset.csv`. Schema: `[id, text, subreddit, label, timestamp]`.

---

## 4. Model Architecture & Normalization Math

CEDV uses **ONE clean, native TensorFlow/Keras architecture** to eliminate Keras 3 / PyTorch friction and ensure fast, stable training within 6GB VRAM bounds:

```
Input (Raw Strings)
       │
       ▼
TextVectorization Layer (max_tokens=25,000, output_sequence_length=128)
       │
       ▼
Embedding Layer (dim=128, mask_zero=True)
       │
       ▼
SpatialDropout1D (0.2)
       │
       ▼
Bidirectional LSTM (64 units, return_sequences=True)
       │
       ▼
Conv1D (128 filters, kernel_size=3, padding='same', activation='relu')
       │
       ▼
GlobalMaxPooling1D
       │
       ▼
Dense (64 units, ReLU) + Dropout (0.3)
       │
       ▼
Dense (8 units, Linear Logits z, dtype='float32')
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Option A: Softmax Simplex                 Option B: Z-Score Standardization
v_i = exp(z_i/T) / sum(exp(z_j/T))        v_i = (z_i - mean(z)) / std(z)
(sum = 1.0, v_i >= 0)                     (mean = 0.0, std = 1.0)
```

### Mathematical Distribution Options

Let $\mathbf{z} = [z_1, z_2, \dots, z_8] \in \mathbb{R}^8$ denote the raw logits output by the projection head:

1. **Option A (Simplex Probability Distribution):**
   $$v_i = \text{Softmax}\left(\frac{z_i}{\tau}\right) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^8 \exp(z_j / \tau)}$$
   - Guaranteed $\sum_{i=1}^8 v_i = 1.0$ and $v_i \ge 0$.
   - Configurable temperature $\tau > 0$ for distribution calibration.

2. **Option B (Standard Normal Distribution / Z-score):**
   $$\mu_{\mathbf{z}} = \frac{1}{8} \sum_{j=1}^8 z_j, \quad \sigma^2_{\mathbf{z}} = \frac{1}{8} \sum_{j=1}^8 (z_j - \mu_{\mathbf{z}})^2$$
   $$v_i = \frac{z_i - \mu_{\mathbf{z}}}{\sqrt{\sigma^2_{\mathbf{z}} + \epsilon}}$$
   - Guaranteed empirical mean $\mu = 0.0$ and unit variance $\sigma = 1.0$.
   - Measures standard deviations above or below the baseline affective state.

---

## 5. Training Pipeline & Hardware Optimization

- **Target Workstation:** NVIDIA GeForce RTX 2060 (6GB VRAM), Ryzen 5 3600 (16GB RAM).
- **Mixed Precision:** `tf.keras.mixed_precision.set_global_policy('mixed_float16')` with float32 logits.
- **Batching & Throughput:** `batch_size = 32`, `max_length = 128`, `tf.data.AUTOTUNE` prefetching.
- **Optimizer & Loss:** `Adam(learning_rate=1e-3)` with `ReduceLROnPlateau` and `CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)`.
- **Peak Memory Budget:** $< 3.5\text{ GB}$ peak VRAM allocation, completely safe from OOM.

---

## 6. Interactive CLI & Live-Test Mode

The interactive terminal application (`cli.py`) supports two primary modes:
1. **Direct Text Analysis:**
   ```bash
   python cli.py --mode direct --text "I feel so exhausted and disconnected from everyone around me."
   ```
   Outputs formatted raw 8-dimensional NumPy vector and a Rich-rendered affective distribution table.
2. **Live Subreddit Fetch:**
   ```bash
   python cli.py --mode live --subreddit anxiety
   ```
   Fetches a live post without API keys via Arctic Shift, displays author and title metadata, and computes instant emotion vector inference.

---

## 7. Step-by-Step Execution Roadmap

- **Phase 1: Keyless Scraping & Dataset Acquisition (`src/scraper.py`)** — Pull recent 2024–2026 posts across the 8 subreddits into `data/dataset.csv`.
- **Phase 2: Preprocessing & Dataset Pipeline (`src/preprocess.py`)** — Text sanitization, 8-class encoding, and `tf.data.Dataset` batching.
- **Phase 3: Model Architecture & Distribution Head (`src/model.py`)** — Native Keras BiLSTM-CNN model and dual-mode vector output head.
- **Phase 4: Training & Evaluation (`src/train.py`)** — Training loop, mixed precision, and performance metrics.
- **Phase 5: Interactive CLI (`cli.py`)** — Rich terminal CLI with direct and live modes.
