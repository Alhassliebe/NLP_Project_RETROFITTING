"""Loaders for embeddings and WordNet lexicons."""
import os, pickle
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn


def load_glove(path: str | Path, vector_size: int | None = None) -> KeyedVectors:
    """Load GloVe text-format embeddings into a gensim KeyedVectors."""
    path = Path(path)
    print(f"Loading GloVe from {path.name}...")
    words, vecs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
    if vector_size is None: vector_size = len(vecs[0])
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(words, np.stack(vecs))
    print(f"  loaded {len(words)} vectors, dim={vector_size}")
    return kv


def build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                          cache_path: str | Path | None = None,
                          lowercase: bool = True) -> dict[str, list[str]]:
    """
    Build the full English WordNet lexicon graph.

    Args:
        relations: subset of {"synonyms", "hypernyms", "hyponyms"}
        cache_path: if given, cache the result to a pickle file
        lowercase: convert all lemmas to lowercase (matches GloVe vocabulary)

    Returns:
        dict mapping each word to its sorted list of related words
    """
    key = tuple(sorted(relations))
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached lexicon from {cache_path}...")
        with open(cache_path, "rb") as f: cached = pickle.load(f)
        if cached.get("relations") == key:
            print(f"  loaded {len(cached['lexicon'])} entries")
            return cached["lexicon"]
        print("  cache mismatch (different relations) — rebuilding")

    print(f"Building WordNet lexicon (relations: {key})...")
    lexicon = {}
    norm = (lambda s: s.lower()) if lowercase else (lambda s: s)
    for synset in wn.all_synsets():
        # collect lemmas of this synset, hypernyms, hyponyms as configured
        out = set()
        if "synonyms" in relations:
            out.update(norm(l.name()) for l in synset.lemmas() if "_" not in l.name())
        if "hypernyms" in relations:
            out.update(norm(l.name()) for hyp in synset.hypernyms() for l in hyp.lemmas() if "_" not in l.name())
        if "hyponyms" in relations:
            out.update(norm(l.name()) for hyp in synset.hyponyms() for l in hyp.lemmas() if "_" not in l.name())
        for lemma in synset.lemmas():
            w = norm(lemma.name())
            if "_" in lemma.name(): continue
            lexicon.setdefault(w, set()).update(out - {w})
    lexicon = {w: sorted(neighbors) for w, neighbors in lexicon.items() if neighbors}
    print(f"  built {len(lexicon)} entries, avg degree {np.mean([len(v) for v in lexicon.values()]):.2f}")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({"relations": key, "lexicon": lexicon}, f)
        print(f"  cached to {cache_path}")
    return lexicon
