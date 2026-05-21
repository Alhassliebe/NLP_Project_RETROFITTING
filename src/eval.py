"""
Evaluation: lexical similarity and sentiment analysis.

Owner: Person C
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from gensim.models import KeyedVectors


def evaluate_similarity(
    embeddings: KeyedVectors,
    dataset_path: Path,
) -> dict[str, float]:
    """
    Evaluate embeddings on a word similarity dataset.

    Args:
        embeddings: KeyedVectors
        dataset_path: CSV with columns [word1, word2, score]

    Returns:
        dict with keys: spearman, pearson, coverage (fraction of pairs evaluable)
    """
    raise NotImplementedError("Person C: implement in Week 1")


def evaluate_sentiment(
    embeddings: KeyedVectors,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate embeddings on a sentiment classification task.

    Strategy: average word vectors per sentence -> logistic regression -> accuracy.

    Returns:
        dict with keys: accuracy, f1
    """
    raise NotImplementedError("Person C: implement in Week 1")


def load_similarity_dataset(name: str) -> pd.DataFrame:
    """Load a similarity benchmark by name."""
    raise NotImplementedError
