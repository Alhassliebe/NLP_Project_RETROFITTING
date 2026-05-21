"""
Central configuration for the retrofitting project.
All paths and shared constants live here so that no module hardcodes paths.
"""
from pathlib import Path

# Project root (resolved relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Embedding files (populate as you download them)
GLOVE_300D_PATH = MODELS_DIR / "glove.6B.300d.txt"
WORD2VEC_PATH = MODELS_DIR / "GoogleNews-vectors-negative300.bin"
FASTTEXT_EN_PATH = MODELS_DIR / "cc.en.300.bin"
FASTTEXT_FR_PATH = MODELS_DIR / "cc.fr.300.bin"

# Lexicon files
PPDB_XL_PATH = DATASETS_DIR / "ppdb-2.0-xl-lexical"
WOLF_PATH = DATASETS_DIR / "wolf-1.0b4.xml"

# Evaluation datasets
RG65_EN_PATH = DATASETS_DIR / "rg65_en.csv"
RG65_FR_PATH = DATASETS_DIR / "rg65_fr.csv"
WORDSIM353_SIM_PATH = DATASETS_DIR / "wordsim353_similarity.csv"
WORDSIM353_REL_PATH = DATASETS_DIR / "wordsim353_relatedness.csv"
SIMLEX999_PATH = DATASETS_DIR / "simlex999.csv"
SST_DIR = DATASETS_DIR / "stanford_sentiment_treebank"

# Retrofitting defaults (from Faruqui et al. 2015)
DEFAULT_N_ITER = 10
DEFAULT_ALPHA = 1.0
# Default beta strategy: 1 / degree(i) — handled inside retrofit()

# Reproducibility
RANDOM_SEED = 42
