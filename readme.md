# Retrofitting Word Vectors to Semantic Lexicons

Reimplementation and analysis of Faruqui et al. (2015),
*Retrofitting Word Vectors to Semantic Lexicons*, NAACL 2015.

## Team

- **Alena** — Core algorithm (`retrofit.py`), OOV strategies, hyperparameter & dimensionality experiments
- **Botakoz** — Data and preprocessing (`preprocessing.py`, `utils.py`), multilingual extension (French)
- **Sharon** — Evaluation (`eval.py`), embedding comparison, sentiment analysis, report coordination

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
| `wolf-1.0b4.xml` | 
| WordNet | via NLTK (auto-downloaded) |
| FrameNet | via NLTK (auto-downloaded) |

### Benchmarks (→ `datasets/`)
```bash
# Download automatically
# downloads RG-65-en, WordSim-353, SimLex-999 python download_benchmarks.py   
# Download manually
# RG-65 French: included in the repository → datasets/rg65_french.txt
# WordSim-353:  curl -L -o datasets/wordsim353crowd.csv "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt"
# SST-2:        downloaded automatically via HuggingFace datasets library
```

---

## Project Structure

```
retrofitting/
├── src/
│   ├── config.py              # All paths and constants
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
│   ├── 03_diagnostics.py               # Alena: centroid drift & edge composition
│   ├── 04_eval_baseline.py             # Sharon: baseline GloVe evaluation
│   ├── 05_retrofit_full_vocab.py       # Alena: full GloVe 300d pipeline, saves .kv
│   ├── 06_oov_comparison.py            # Alena: intersection vs filtering vs mean_synonyms
│   ├── 07_convergence_analysis.py      # Alena: n_iter sensitivity
│   │                                   #   → figures/convergence.png
│   ├── 08_dimensionality_experiment.py # Alena: 50/100/200/300d comparison
│   │                                   #   → figures/dimensionality.png
│   │                                   #   → figures/dimensionality_delta.png
│   ├── 08_french_experiment.py         # Botakoz: early French prototype (standalone)
│   ├── 09_alpha_beta_grid.py           # Alena: hyperparameter sensitivity (α, β)
│   │                                   #   - α sweep (β=inv_degree, 10 steps)
│   │                                   #   - β sweep (α=1.0, 10 steps)
│   │                                   #   - 2D joint grid (10×10 = 100 runs)
│   │                                   #   → figures/alpha_beta_curves.png
│   │                                   #   → figures/alpha_beta_heatmap.png
│   ├── 10_french_experiment.py         # Botakoz: fastText-fr + Wolf + RG-65-fr
│   ├── 11_sentiment_analysis.py        # Sharon: SST-2 baseline vs retrofitted
│   ├── 12_qualitative_analysis.py      # Sharon: top-15 pairs improved/worsened
│   ├── 13_optimization_benchmark.py    # Alena: runtime profiling of retrofit loop
│   └── 14_convergence_prototype.py     # Alena: n_iter convergence on prototype sample
├── datasets/                   # Benchmark datasets
│   ├── rg65_en.csv             # RG-65 English
│   ├── rg65_french.txt         # RG-65 French
│   ├── simlex999.csv           # SimLex-999
│   └── wordsim353crowd.csv     # WordSim-353
├── models/                                # Pre-trained embeddings (large files, not in git)
│   ├── glove.6B.{50,100,200,300}d.txt
│   ├── glove_300d_retrofitted_wn_all.kv   # saved by notebook 05
│   └── wn_all_lexicon.pkl                 # cached by preprocessing
├── figures/                       # Generated plots
│   ├── alpha_beta_curves.png      # sweep line plots (notebook 09)
│   ├── alpha_beta_heatmap.png     # 2D grid heatmaps (notebook 09)
│   ├── convergence.png            # n_iter convergence (notebook 07)
│   ├── dimensionality.png         # Spearman ρ by embedding size (notebook 08)
│   └── dimensionality_delta.png   # Δ Spearman ρ by embedding size (notebook 08)
├── results/                             # CSV outputs from experiments
│   ├── alpha_beta_grid.csv              # earlier grid search results
│   ├── alpha_sweep.csv                  # α sweep (notebook 09)
│   ├── beta_sweep.csv                   # β sweep (notebook 09)
│   ├── grid2d.csv                       # 2D joint grid (notebook 09)
│   ├── convergence_analysis.csv         # notebook 07
│   ├── convergence_prototype.csv        # notebook 14
│   ├── dimensionality_experiment.csv    # notebook 08
│   ├── oov_comparison.csv               # notebook 06
│   ├── optimization_benchmark.csv       # notebook 13
│   ├── qualitative_full.csv             # notebook 12
│   ├── qualitative_top_decrease.csv     # notebook 12
│   ├── qualitative_top_increase.csv     # notebook 12
│   └── sentiment_analysis.csv           # notebook 11
├── main.py                    # CLI entry point
├── download_benchmarks.py     # Auto-download RG-65, SimLex-999
├── fix_wordsim.py             # Creates wordsim353_similarity.csv if missing
├── test_eval.py               # Quick sanity check for eval pipeline
├── requirements.txt
└── readme.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Alhassliebe/NLP_Project_RETROFITTING.git
cd NLP_Project_RETROFITTING

python3 -m venv venv

source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install datasets

python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('framenet_v17')"
```

### 2. Download large files

These files are too large for Git and must be downloaded manually.

**English embeddings — place in `models/`**

GloVe 6B (all four sizes, ~820 MB total):
```bash
# macOS / Linux
curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d models/
rm glove.6B.zip
```
```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri https://nlp.stanford.edu/data/glove.6B.zip -OutFile glove.6B.zip
Expand-Archive glove.6B.zip -DestinationPath models/
Remove-Item glove.6B.zip
```

**French embeddings — place in `models/`** (required for notebook 10 only, ~4 GB):
```bash
# macOS / Linux
curl -L -o models/cc.fr.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz
gunzip models/cc.fr.300.bin.gz
```
```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz -OutFile models\cc.fr.300.bin.gz
# then decompress with 7-Zip or another tool
```

**French WordNet — place in `datasets/`** (required for notebook 10 only):

Download `wolf-1.0b4.xml` from https://gforge.inria.fr/frs/?group_id=5631 and place it at `datasets/wolf-1.0b4.xml`.

**English benchmarks** (RG-65, SimLex-999, WordSim-353) are already included in the repository under `datasets/`.

### 3. Run individual experiments

Run each notebook from the **project root** (not from inside `notebooks/`). Run them one at a time in a fresh terminal:

```bash
cd NLP_Project_RETROFITTING

source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

python notebooks/01_retrofit_prototype.py
```

```bash
# Prototype on 212-word sample
python notebooks/01_retrofit_prototype.py

# Sanity check: production vs prototype
python notebooks/02_test_retrofit.py

# Centroid drift & lexicon diagnostics
python notebooks/03_diagnostics.py

# Baseline evaluation (no retrofitting)
python notebooks/04_eval_baseline.py

# Full GloVe 300d retrofit — saves retrofitted vectors to models/
python notebooks/05_retrofit_full_vocab.py

# OOV strategy comparison
python notebooks/06_oov_comparison.py

# n_iter sensitivity → figures/convergence.png
python notebooks/07_convergence_analysis.py

# Dimensionality experiment (50/100/200/300d) → figures/dimensionality*.png
python notebooks/08_dimensionality_experiment.py

# Hyperparameter sensitivity (α/β sweeps + 2D grid) → figures/alpha_beta_*.png
python notebooks/09_alpha_beta_grid.py

# French experiment (fastText-fr + Wolf)
python notebooks/10_french_experiment.py

# Sentiment analysis (SST-2)
python notebooks/11_sentiment_analysis.py

# Qualitative analysis — requires models/glove_300d_retrofitted_wn_all.kv
python notebooks/12_qualitative_analysis.py

# Runtime profiling of retrofit loop
python notebooks/13_optimization_benchmark.py

# Convergence on prototype sample
python notebooks/14_convergence_prototype.py
```

**Important notes:**
- Notebooks 05 and 09 each take 30–60 minutes (full GloVe 300d).
- Notebook 12 requires notebook 05 to have run first (it loads the saved retrofitted vectors from `models/`).
- Notebooks 10, 11, 12 require `cc.fr.300.bin` and `wolf-1.0b4.xml`. All other notebooks run on English data only and have no additional dependencies beyond `models/glove.6B.*.txt`.
- SST-2 data for notebook 11 is downloaded automatically on first run via the HuggingFace `datasets` library.

### 4. Run the pipeline

**CLI (single configuration):**

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

---

## Key Results

### English (GloVe 300d + WN_all, intersection OOV, n_iter=10)

| Benchmark | Baseline ρ | Retrofitted ρ | Δ |
|---|---|---|---|
| RG-65 | 0.766 | 0.843 | +0.077 |
| SimLex-999 | 0.371 | 0.461 | +0.091 |
| WordSim-353 | 0.543 | 0.606 | +0.063 |

### French (fastText-fr + Wolf, n_iter=10)

| Benchmark | Baseline ρ | Retrofitted ρ | Δ |
|---|---|---|---|
| RG-65-fr | 0.8135 | 0.791 | -0.0196 |

### Sentiment Analysis (SST-2, GloVe 300d + WN_all)

| Metric | Baseline | Retrofitted | Δ |
|---|---|---|---|
| Accuracy | 0.7718 | 0.7683 | -0.0035 |
| F1 | 0.7858 | 0.7837 | -0.0021 |

---

## References

Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015).
*Retrofitting Word Vectors to Semantic Lexicons*. NAACL 2015.
[arXiv:1411.4166](https://arxiv.org/abs/1411.4166)
