# Retrofitting Word Vectors to Semantic Lexicons

Reimplementation and analysis of Faruqui et al. (2015),
*Retrofitting Word Vectors to Semantic Lexicons*, NAACL 2015.

## Team

- **Alena** — Core algorithm (`retrofit.py`), OOV strategies, hyperparameter & dimensionality experiments
- **Botakoz** — Data and preprocessing (`preprocessing.py`, `utils.py`), multilingual extension (French)
- **Sharon** — Evaluation (`eval.py`), embedding comparison, sentiment analysis, report coordination

---

## Setup

```bash
# Clone
git clone <repo-url>
cd retrofitting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
pip install datasets           # for sentiment analysis (SST-2)

# Download NLTK resources
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('framenet_v17')"
```

---

## Data

Download manually into `models/` and `datasets/`:

### Embeddings (→ `models/`)
| File | Size | Source |
|---|---|---|
| `glove.6B.300d.txt` | ~1 GB | https://nlp.stanford.edu/data/glove.6B.zip |
| `cc.fr.300.bin` | ~4 GB | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz |

Note: Word2Vec Google News was excluded due to memory constraints (8 GB RAM).

### Lexicons (→ `datasets/`)
| File | Source |
|---|---|
| `wolf-1.0b4.xml` | https://almanach.inria.fr (or Google Drive link in task brief) |
| WordNet | via NLTK (auto-downloaded) |
| FrameNet | via NLTK (auto-downloaded) |

### Benchmarks (→ `datasets/`)
```bash
# Download automatically
python download_benchmarks.py   # downloads RG-65-en, SimLex-999

# Download manually
# RG-65 French: Google Drive link in task brief → datasets/rg65_french.txt
# WordSim-353:  curl -L -o datasets/wordsim353crowd.csv "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt"
# SST-2:        downloaded automatically via HuggingFace datasets library
```

---

## Project Structure

```
retrofitting/
├── src/
│   ├── config.py              # All paths and constants (edit here if files move)
│   ├── retrofit.py            # Core retrofitting algorithm (Alena)
│   │                          #   - retrofit(): main update loop (eq. 4, Faruqui 2015)
│   │                          #   - _resolve_oov(): OOV strategy dispatcher
│   │                          #   - _beta(): neighbor weight strategies
│   │                          #   - cosine_similarity()
│   ├── preprocessing.py       # Data loaders (Botakoz)
│   │                          #   - load_glove(): GloVe text format → KeyedVectors
│   │                          #   - load_fasttext(): fastText .bin → KeyedVectors
│   │                          #   - build_wordnet_lexicon(): WN_syn / WN_all / hyper / hypo
│   │                          #   - build_wolf_lexicon(): French WordNet XML parser
│   │                          #   - build_framenet_lexicon(): FrameNet co-frame graph
│   │                          #   - build_lexicon(name): unified dispatcher
│   ├── eval.py                # Evaluation (Sharon)
│   │                          #   - evaluate_similarity(): Spearman ρ on word pairs
│   │                          #   - evaluate_all(): run all benchmarks at once
│   │                          #   - evaluate_sentiment(): SST logistic regression
│   └── utils.py               # Shared utilities (Botakoz)
│                              #   - setup_logging()
│                              #   - vocab_overlap()
├── notebooks/
│   ├── 01_retrofit_prototype.py        # Alena: prototype on 212-word sample
│   ├── 02_test_retrofit.py             # Alena: sanity check production vs prototype
│   ├── 03_diagnostics.py               # Alena: centroid drift analysis
│   ├── 04_eval_baseline.py             # Alena/Sharon: baseline GloVe evaluation
│   ├── 05_retrofit_full_vocab.py       # Alena: full GloVe 300d pipeline
│   ├── 06_oov_comparison.py            # Alena: OOV strategy comparison
│   ├── 07_convergence_analysis.py      # Alena: n_iter sensitivity
│   ├── 08_dimensionality_experiment.py # Alena: 50/100/200/300d comparison
│   ├── 09_alpha_beta_grid.py           # Alena: hyperparameter sensitivity
│   │                                   #   - α sweep (β fixed at inv_degree, 10 steps)
│   │                                   #   - β sweep (α fixed at 1.0, 10 steps)
│   │                                   #   - 2D joint grid search (10×10 = 100 runs)
│   │                                   #   → figures/alpha_beta_curves.png
│   │                                   #   → figures/alpha_beta_heatmap.png
│   ├── 10_french_experiment.py         # Botakoz: fastText-fr + Wolf + RG-65-fr
│   ├── 11_sentiment_analysis.py        # Sharon: SST-2 baseline vs retrofitted
│   ├── 12_qualitative_analysis.py      # Sharon: top-15 pairs improved/worsened
│   ├── 13_optimization_benchmark.py    # Alena: runtime profiling of retrofit loop
│   └── 14_convergence_prototype.py     # Alena: convergence behaviour prototype
├── datasets/                  # Benchmark datasets (gitignored)
├── models/                    # Pre-trained embeddings (gitignored)
├── figures/                   # Generated plots (gitignored)
│   ├── alpha_beta_curves.png  #   sweep line plots (notebook 09)
│   └── alpha_beta_heatmap.png #   2D grid heatmaps (notebook 09)
├── results/                   # CSV outputs from experiments (gitignored)
│   ├── alpha_sweep.csv        #   α sweep results
│   ├── beta_sweep.csv         #   β sweep results
│   └── grid2d.csv             #   2D grid results
├── main.py                    # CLI entry point
├── download_benchmarks.py     # Auto-download RG-65, SimLex-999
├── requirements.txt
└── readme.md
```

---

## Usage

### Run the full pipeline

```bash
# English: GloVe 300d + WordNet WN_all
python main.py --embedding glove --lexicon wn_all --benchmark rg65 simlex999 wordsim353

# English: GloVe 300d + WordNet WN_syn only
python main.py --embedding glove --lexicon wn_syn --benchmark rg65 simlex999

# French: fastText-fr + Wolf
python main.py --embedding fasttext --lexicon wolf --benchmark rg65

# Baseline only (no retrofitting)
python main.py --embedding glove --lexicon wn_all --no-retrofit
```

### Arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--embedding` | `glove`, `fasttext` | required | Pre-trained embeddings |
| `--lexicon` | `wn_syn`, `wn_all`, `wn_hyper`, `wn_hypo`, `framenet`, `wolf` | required | Semantic lexicon |
| `--benchmark` | `rg65`, `simlex999`, `wordsim353` | all three | Evaluation benchmarks |
| `--n-iter` | integer | 10 | Number of retrofit iterations |
| `--alpha` | float | 1.0 | Weight of original vector |
| `--beta` | `inv_degree`, `uniform`, `inv_sq_degree` | `inv_degree` | Neighbor weight strategy |
| `--oov-strategy` | `intersection`, `filtering`, `mean_synonyms` | `intersection` | OOV handling |
| `--no-retrofit` | flag | False | Evaluate baseline only |

Retrofitted vectors are automatically saved to `models/` after each run.

### Run individual experiments

```bash
# Hyperparameter sensitivity (α/β sweeps + 2D grid) — saves 3 CSVs and 2 figures
python notebooks/09_alpha_beta_grid.py

# French multilingual experiment
python notebooks/10_french_experiment.py

# Sentiment analysis (SST-2)
python notebooks/11_sentiment_analysis.py

# Qualitative analysis (top pairs improved/worsened)
python notebooks/12_qualitative_analysis.py

# Runtime profiling
python notebooks/13_optimization_benchmark.py

# Convergence behaviour
python notebooks/14_convergence_prototype.py
```

---

## Key Results

### English (GloVe 300d + WN_all, intersection OOV, n_iter=10)

| Benchmark | Baseline ρ | Retrofitted ρ | Δ |
|---|---|---|---|
| RG-65 | 0.766 | 0.843 | +0.077 |
| SimLex-999 | 0.371 | 0.461 | +0.091 |
| WordSim-353 | 0.609 | 0.624 | +0.015 |

### French (fastText-fr + Wolf, intersection OOV, n_iter=10)

| Benchmark | Baseline ρ | Retrofitted ρ | Δ |
|---|---|---|---|
| RG-65-fr | 0.811 | 0.791 | -0.020 |

### Sentiment Analysis (SST-2, GloVe 300d + WN_all)

| Metric | Baseline | Retrofitted | Δ |
|---|---|---|---|
| Accuracy | 0.7706 | 0.7672 | -0.0034 |
| F1 | 0.7845 | 0.7824 | -0.0021 |

---

## References

Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015).
*Retrofitting Word Vectors to Semantic Lexicons*. NAACL 2015.
[arXiv:1411.4166](https://arxiv.org/abs/1411.4166)
