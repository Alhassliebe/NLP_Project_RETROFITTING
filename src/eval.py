"""Evaluation: similarity benchmarks (Spearman ρ) and sentiment classification."""
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import KeyedVectors


# benchmark file paths and column names
# WordSim-353 can be the split (similarity) or crowd version, try both
_LOADERS = {
    "rg65":       (["datasets/rg65_en.csv"], ",", "word1", "word2", "score"),
    "simlex999":  (["datasets/simlex999.csv"], ",", "word1", "word2", "score"),
    "wordsim353": (
        ["datasets/wordsim353_similarity.csv", "datasets/wordsim353crowd.csv"],
        ",",
        ("word1", "Word 1"),
        ("word2", "Word 2"),
        ("score", "Human (Mean)"),
    ),
}


def _resolve_column(df: pd.DataFrame, candidates) -> str:
    """Pick whichever candidate column name exists in df."""
    if isinstance(candidates, str):
        return candidates
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


def load_benchmark(name: str) -> pd.DataFrame:
    """Load a similarity benchmark by short name → columns [word1, word2, score]."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown benchmark: {name}. Known: {list(_LOADERS)}")
    paths, sep, c1, c2, cs = _LOADERS[name]
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"None of these benchmark files found: {paths}")
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    col1, col2, cols = _resolve_column(df, c1), _resolve_column(df, c2), _resolve_column(df, cs)
    return df.rename(columns={col1: "word1", col2: "word2", cols: "score"})[["word1", "word2", "score"]]


def _cos(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def evaluate_similarity(embeddings: KeyedVectors, benchmark: str) -> dict:
    """Spearman ρ between embedding cosine similarity and human judgments."""
    df = load_benchmark(benchmark)
    df["word1"], df["word2"] = df["word1"].str.lower(), df["word2"].str.lower()
    vocab = embeddings.key_to_index
    keep = df[df["word1"].isin(vocab) & df["word2"].isin(vocab)].copy()
    keep["cos"] = [_cos(embeddings[r.word1], embeddings[r.word2]) for r in keep.itertuples()]
    rho, p = spearmanr(keep["cos"], keep["score"])
    return {
        "benchmark":    benchmark,
        "n_total":      len(df),
        "n_evaluated":  len(keep),
        "coverage":     round(len(keep) / len(df), 4),
        "spearman_rho": round(float(rho), 4),
        "p_value":      round(float(p), 6),
    }


def evaluate_all(embeddings: KeyedVectors, benchmarks: list[str] | None = None) -> pd.DataFrame:
    """Evaluate on all benchmarks; return a tidy DataFrame."""
    benchmarks = benchmarks or list(_LOADERS.keys())
    return pd.DataFrame([evaluate_similarity(embeddings, b) for b in benchmarks])


def evaluate_sentiment(
    embeddings: KeyedVectors,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, float]:
    """Sentence-level sentiment: average word vectors → logistic regression on SST-2.

    train_data / test_data: DataFrame with columns
        tokens: list[str]  (tokenized, lowercased)
        label:  int        (0 = negative, 1 = positive)
    """
    def sentence_vec(tokens):
        vecs = [embeddings[w] for w in tokens if w in embeddings]
        return np.mean(vecs, axis=0) if vecs else None

    def build_XY(df):
        X, y = [], []
        for _, row in df.iterrows():
            vec = sentence_vec(row["tokens"])
            if vec is not None:
                X.append(vec); y.append(int(row["label"]))
        return np.array(X), np.array(y)

    X_train, y_train = build_XY(train_data)
    X_test,  y_test  = build_XY(test_data)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds, average="binary"), 4),
    }
