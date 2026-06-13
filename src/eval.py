"""
Evaluation: lexical similarity benchmarks and sentiment analysis.
Owner: Person C
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import KeyedVectors

_LOADERS = {
    "rg65":       ("datasets/rg65_en.csv",              ",", "word1", "word2", "score"),
    "simlex999":  ("datasets/simlex999.csv",             ",", "word1", "word2", "score"),
    "wordsim353": ("datasets/wordsim353_similarity.csv", ",", "word1", "word2", "score"),
}

def load_benchmark(name: str) -> pd.DataFrame:
    if name not in _LOADERS:
        raise ValueError(f"Unknown benchmark: {name}. Known: {list(_LOADERS)}")
    path, sep, c1, c2, cs = _LOADERS[name]
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={c1: "word1", c2: "word2", cs: "score"})[["word1", "word2", "score"]]

def _cos(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def evaluate_similarity(embeddings: KeyedVectors, benchmark: str) -> dict:
    """Spearman correlation between cosine similarity and human judgments."""
    df = load_benchmark(benchmark)
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
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
        "p_value":      round(float(p), 4),
    }

def evaluate_all(embeddings: KeyedVectors, benchmarks: list[str] = None) -> pd.DataFrame:
    """Evaluate on all benchmarks; return a DataFrame summary."""
    benchmarks = benchmarks or list(_LOADERS.keys())
    return pd.DataFrame([evaluate_similarity(embeddings, b) for b in benchmarks])

def evaluate_sentiment(
    embeddings: KeyedVectors,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Average word vectors per sentence -> logistic regression -> accuracy + f1.
    train_data / test_data: DataFrame with columns [tokens, label]
        tokens: list of str (already tokenized and lowercased)
        label:  0 (negative) or 1 (positive)
    """
    def sentence_vec(tokens):
        vecs = [embeddings[w] for w in tokens if w in embeddings]
        return np.mean(vecs, axis=0) if vecs else None

    def build_XY(df):
        X, y = [], []
        for _, row in df.iterrows():
            vec = sentence_vec(row["tokens"])
            if vec is not None:
                X.append(vec)
                y.append(int(row["label"]))
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