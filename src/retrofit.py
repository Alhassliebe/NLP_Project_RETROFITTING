"""Retrofitting algorithm (Faruqui et al., 2015). Alena Muravyeva - Week 1, Tasks A1.3-A1.4."""
import numpy as np
from gensim.models import KeyedVectors


def _beta(degree: int, strategy: str) -> float:
    """Neighbor weight beta_ij given the strategy and degree(i)."""
    if strategy == "inv_degree":    return 1.0 / degree
    if strategy == "uniform":       return 1.0
    if strategy == "inv_sq_degree": return 1.0 / (degree ** 2)
    raise ValueError(f"Unknown beta strategy: {strategy}")


def retrofit(word_vecs: KeyedVectors, lexicon: dict[str, list[str]],
             n_iter: int = 10, alpha: float = 1.0, beta: str = "inv_degree",
             return_convergence: bool = False):
    """
    Retrofit embeddings using a semantic lexicon (Faruqui et al., 2015, eq. 4):
        q_i = (alpha * q_hat_i + sum_j beta_ij * q_j) / (alpha + sum_j beta_ij)

    Args:
        word_vecs: pre-trained embeddings (not modified)
        lexicon: word -> list of related words (intersection OOV strategy applied here)
        n_iter: number of update iterations
        alpha: weight of the original vector
        beta: neighbor weight strategy ("inv_degree" | "uniform" | "inv_sq_degree")
        return_convergence: if True, also return per-iteration L2 change

    Returns:
        new KeyedVectors with retrofitted vectors (and convergence log if requested)
    """
    # Working vocab: words present in both embeddings and lexicon (intersection OOV)
    vocab = [w for w in lexicon if w in word_vecs.key_to_index]
    if not vocab: raise ValueError("No overlap between embeddings and lexicon")

    idx = {w: i for i, w in enumerate(vocab)}
    original = np.stack([word_vecs[w] for w in vocab]).astype(np.float32)
    new = original.copy()
    neighbor_idx = [np.array([idx[n] for n in lexicon[w] if n in idx], dtype=np.int64) for w in vocab]
    beta_vals = np.array([_beta(len(ni), beta) if len(ni) else 0.0 for ni in neighbor_idx], dtype=np.float32)

    log = []
    for it in range(n_iter):
        prev = new.copy()
        for i, neighbors in enumerate(neighbor_idx):
            if len(neighbors) == 0: continue
            b = beta_vals[i]
            new[i] = (alpha * original[i] + b * new[neighbors].sum(axis=0)) / (alpha + b * len(neighbors))
        log.append(float(np.linalg.norm(new - prev)))

    # Preserve full vocabulary, overwrite retrofitted rows
    out = KeyedVectors(vector_size=word_vecs.vector_size)
    out.add_vectors(list(word_vecs.key_to_index.keys()), word_vecs.vectors)
    for w, i in idx.items(): out[w] = new[i]

    return (out, log) if return_convergence else out


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
