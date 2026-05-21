# Retrofitting Word Vectors to Semantic Lexicons

Reimplementation and analysis of Faruqui et al. (2015), *Retrofitting Word Vectors to Semantic Lexicons*.

## Team

- **Person A** — Core algorithm (`retrofit.py`), OOV strategies, hyperparameter experiments
- **Person B** — Data and preprocessing (`preprocessing.py`, `utils.py`), multilingual extension
- **Person C** — Evaluation (`eval.py`), embedding comparison, report coordination

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

# Download NLTK resources
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('framenet_v17')"
```

## Data

Download manually into `models/` and `datasets/` (see `src/config.py` for expected paths):

- **Embeddings:** GloVe 300d, Word2Vec Google News, fastText (en, fr)
- **Lexicons:** WordNet (via NLTK), PPDB XL lexical, Wolf (French WordNet), FrameNet (via NLTK)
- **Benchmarks:** RG-65, WordSim-353, SimLex-999, Stanford Sentiment Treebank

## Project structure

```
retrofitting/
├── src/
│   ├── config.py           # All paths and constants
│   ├── retrofit.py         # Core algorithm (Person A)
│   ├── preprocessing.py    # Loaders for embeddings and lexicons (Person B)
│   ├── eval.py             # Evaluation benchmarks (Person C)
│   └── utils.py            # Shared utilities
├── datasets/               # Benchmark datasets (gitignored)
├── models/                 # Pre-trained embeddings (gitignored)
├── figures/                # Generated plots (gitignored)
├── notebooks/              # Exploratory notebooks
├── main.py                 # CLI entry point
├── requirements.txt
└── readme.md
```

## Usage

```bash
python main.py --embedding glove --lexicon wn_all --benchmark rg65 simlex999
```

## Git workflow

- `main` — protected, only PR merges
- Feature branches: `feature/algorithm` (A), `feature/data` (B), `feature/evaluation` (C)
- Open a PR for each task. Reviewer = the other two; one approval is enough
- No force-pushes to `main`. Squash-and-merge preferred

## References

Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). *Retrofitting Word Vectors to Semantic Lexicons*. NAACL 2015. [arXiv:1411.4166](https://arxiv.org/abs/1411.4166)
