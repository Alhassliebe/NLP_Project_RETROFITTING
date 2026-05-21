"""
Retrofitting algorithm (Faruqui et al., 2015).

Owner: Person A
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from gensim.models import KeyedVectors


def retrofit(
    word_vecs: KeyedVectors,
    lexicon: dict[str, list[str]],
    n_iter: int = 10,
    alpha: float = 1.0,
    beta: Optional[str] = None,
) -> KeyedVectors:
    """
    Apply retrofitting to refine pre-trained embeddings using a semantic lexicon.

    Update rule (Faruqui et al., 2015, eq. 4):
        q_i = (alpha_i * q_hat_i + sum_j beta_ij * q_j) / (alpha_i + sum_j beta_ij)

    Args:
        word_vecs: pre-trained embeddings (will not be modified in place)
        lexicon: word -> list of related words
        n_iter: number of iterations
        alpha: weight of the original vector (default 1.0)
        beta: strategy for neighbor weights. None -> 1/degree(i). Other options:
              "uniform", "inv_sq_degree" (1/degree(i)^2)

    Returns:
        new KeyedVectors with retrofitted embeddings
    """
    raise NotImplementedError("Person A: implement in Week 1")


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    raise NotImplementedError
