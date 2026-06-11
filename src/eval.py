"""Evaluation: lexical similarity benchmarks (Spearman correlation with human judgments)."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors


# Loaders cope with the slight format differences across our benchmark CSVs.
# All return a DataFrame with columns: word1, word2, score.
_LOADERS = {
    "rg65":      ("datasets/rg65_en.csv",        ",",  "word1", "word2", "score"),
    "simlex999": ("datasets/simlex999.csv",      ",",  "word1", "word2", "score"),
    "wordsim353": ("datasets/wordsim353crowd.csv", ",", "Word 1", "Word 2", "Human (Mean)"),
}


def load_benchmark(name: str) -> pd.DataFrame:
    """Load a lexical similarity benchmark by short name."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown benchmark: {name}. Known: {list(_LOADERS)}")
    path, sep, c1, c2, cs = _LOADERS[name]
    df = pd.read_csv(path, sep=sep)
    return df.rename(columns={c1: "word1", c2: "word2", cs: "score"})[["word1", "word2", "score"]]


def _cos(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def evaluate_similarity(embeddings: KeyedVectors, benchmark: str) -> dict:
    """
    Evaluate embeddings on a similarity benchmark via Spearman rank correlation
    between cosine similarity and human judgments. Pairs with OOV words are skipped.
    """
    df = load_benchmark(benchmark)
    df["word1"], df["word2"] = df["word1"].str.lower(), df["word2"].str.lower()
    vocab = embeddings.key_to_index
    keep = df[df["word1"].isin(vocab) & df["word2"].isin(vocab)].copy()
    keep["cos"] = [_cos(embeddings[r.word1], embeddings[r.word2]) for r in keep.itertuples()]
    rho, p = spearmanr(keep["cos"], keep["score"])
    return {"benchmark": benchmark, "n_total": len(df), "n_evaluated": len(keep),
            "coverage": len(keep) / len(df), "spearman_rho": float(rho), "p_value": float(p)}


def evaluate_all(embeddings: KeyedVectors, benchmarks: list[str] = None) -> pd.DataFrame:
    """Evaluate on a list of benchmarks; return a DataFrame summary."""
    benchmarks = benchmarks or list(_LOADERS.keys())
    return pd.DataFrame([evaluate_similarity(embeddings, b) for b in benchmarks])
