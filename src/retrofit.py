"""Retrofitting algorithm (Faruqui et al., 2015)."""
import numpy as np
from gensim.models import KeyedVectors


def _beta(degree: int, strategy) -> float:
    """Compute beta_ij given the strategy and degree(i).

    strategy can be a float (direct uniform weight) or one of:
      "inv_degree" (1/degree, paper default), "uniform" (1.0), "inv_sq_degree" (1/degree²)
    """
    if isinstance(strategy, (int, float)):
        return float(strategy)
    if strategy == "inv_degree":    return 1.0 / degree
    if strategy == "uniform":       return 1.0
    if strategy == "inv_sq_degree": return 1.0 / (degree ** 2)
    raise ValueError(f"Unknown beta strategy: {strategy}")


def _resolve_oov(word_vecs: KeyedVectors, lexicon: dict[str, list[str]],
                 oov_strategy: str, fasttext_model=None):
    """Return (vocab, vectors, lexicon) under the chosen OOV strategy."""
    emb_set = set(word_vecs.key_to_index)
    lex_set = set(lexicon.keys()) | {n for ns in lexicon.values() for n in ns}

    if oov_strategy == "intersection":
        vocab = sorted(emb_set & lex_set)
        vecs = np.stack([word_vecs[w] for w in vocab]).astype(np.float32)
        vocab_set = set(vocab)
        new_lex = {w: [n for n in lexicon.get(w, []) if n in vocab_set] for w in vocab}

    elif oov_strategy == "filtering":
        vocab = sorted(emb_set)
        vecs = np.stack([word_vecs[w] for w in vocab]).astype(np.float32)
        new_lex = {w: [n for n in lexicon.get(w, []) if n in emb_set]
                   for w in vocab if w in lexicon}

    elif oov_strategy == "mean_synonyms":
        synth = {}
        for w, neighbours in lexicon.items():
            if w in emb_set: continue
            known = [n for n in neighbours if n in emb_set]
            if known:
                synth[w] = np.mean([word_vecs[n] for n in known], axis=0).astype(np.float32)
        vocab = sorted(emb_set | set(synth.keys()))
        idx = {w: i for i, w in enumerate(vocab)}
        vecs = np.zeros((len(vocab), word_vecs.vector_size), dtype=np.float32)
        for w in vocab:
            vecs[idx[w]] = synth[w] if w in synth else word_vecs[w]
        new_lex = {w: [n for n in lexicon.get(w, []) if n in idx]
                   for w in vocab if w in lexicon}

    elif oov_strategy == "fasttext":
        if fasttext_model is None:
            raise ValueError("fasttext strategy requires fasttext_model argument")
        synth = {}
        for w in lex_set - emb_set:
            try:
                synth[w] = fasttext_model.get_word_vector(w).astype(np.float32)
            except Exception:
                pass
        vocab = sorted(emb_set | set(synth.keys()))
        idx = {w: i for i, w in enumerate(vocab)}
        vecs = np.zeros((len(vocab), word_vecs.vector_size), dtype=np.float32)
        for w in vocab:
            vecs[idx[w]] = synth[w] if w in synth else word_vecs[w]
        new_lex = {w: [n for n in lexicon.get(w, []) if n in idx]
                   for w in vocab if w in lexicon}

    else:
        raise ValueError(f"Unknown OOV strategy: {oov_strategy}")

    return vocab, vecs, new_lex


def retrofit(word_vecs: KeyedVectors, lexicon: dict[str, list[str]],
             n_iter: int = 10, alpha: float = 1.0, beta: str | float = "inv_degree",
             oov_strategy: str = "intersection", fasttext_model=None,
             return_convergence: bool = False, verbose: bool = False):
    """Retrofit word embeddings using a semantic lexicon (Faruqui et al. 2015, eq. 4).

    alpha: weight of the original vector (paper default = 1.0)
    beta: per-neighbor weight — float for uniform or string strategy
          ("inv_degree" | "uniform" | "inv_sq_degree")
    oov_strategy: how to handle words missing from one side
                  ("intersection" | "filtering" | "mean_synonyms" | "fasttext")
    """
    vocab, original, lex = _resolve_oov(word_vecs, lexicon, oov_strategy, fasttext_model)
    if not vocab:
        raise ValueError("Empty vocabulary after OOV resolution")

    idx = {w: i for i, w in enumerate(vocab)}
    new = original.copy()
    neighbor_idx = [
        np.array([idx[n] for n in lex.get(w, []) if n in idx], dtype=np.int64)
        for w in vocab
    ]
    beta_vals = np.array(
        [_beta(len(ni), beta) if len(ni) else 0.0 for ni in neighbor_idx],
        dtype=np.float32,
    )

    if verbose:
        with_nbrs = sum(1 for ni in neighbor_idx if len(ni) > 0)
        print(f"  OOV={oov_strategy}: |vocab|={len(vocab)}, words with neighbours={with_nbrs}")

    log = []
    for it in range(n_iter):
        prev = new.copy()
        for i, neighbors in enumerate(neighbor_idx):
            if len(neighbors) == 0:
                continue
            b = beta_vals[i]
            new[i] = (alpha * original[i] + b * new[neighbors].sum(axis=0)) / (alpha + b * len(neighbors))
        change = float(np.linalg.norm(new - prev))
        log.append(change)
        if verbose:
            print(f"  iter {it+1:2d}: L2 change = {change:.4f}")

    # build output KeyedVectors: start from original matrix, overwrite retrofitted rows
    if verbose:
        print("  assembling output KeyedVectors...")
    all_keys = list(word_vecs.key_to_index.keys())
    n_orig = len(all_keys)
    extra_keys = [w for w in vocab if w not in word_vecs.key_to_index]
    full_keys = all_keys + extra_keys
    full_vecs = np.empty((len(full_keys), word_vecs.vector_size), dtype=np.float32)
    full_vecs[:n_orig] = word_vecs.vectors
    orig_idx = word_vecs.key_to_index
    for w, i in idx.items():
        if w in orig_idx:
            full_vecs[orig_idx[w]] = new[i]
    for j, w in enumerate(extra_keys):
        full_vecs[n_orig + j] = new[idx[w]]
    out = KeyedVectors(vector_size=word_vecs.vector_size)
    out.add_vectors(full_keys, full_vecs)

    return (out, log) if return_convergence else out


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
