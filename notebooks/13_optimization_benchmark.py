"""Benchmark of the two retrofit optimisations on the 212-word prototype sample."""
import random, sys, time
import numpy as np
from copy import deepcopy
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn

GLOVE_PATH = "models/glove.6B.100d.txt"
SEED = 42
N_SEEDS_NOUN, N_SEEDS_ADJ = 25, 25
NEIGHBORS_PER_SEED = 4
TOP_K_GLOVE = 20000
N_RUNS = 5
N_ITER = 10


def get_related(word, relations):
    out = set()
    for s in wn.synsets(word):
        if "synonyms" in relations:
            out.update(l.name().lower() for l in s.lemmas() if "_" not in l.name())
        if "hypernyms" in relations:
            out.update(l.name().lower() for hyp in s.hypernyms() for l in hyp.lemmas() if "_" not in l.name())
        if "hyponyms" in relations:
            out.update(l.name().lower() for hyp in s.hyponyms() for l in hyp.lemmas() if "_" not in l.name())
    return out - {word}


def collect_candidates(pos):
    cands = []
    for synset in wn.all_synsets(pos=pos):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if "_" in name: continue
            syns = {l.name().lower() for s in wn.synsets(name, pos=pos) for l in s.lemmas()
                    if "_" not in l.name() and l.name().lower() != name}
            if len(syns) < 2: continue
            if pos == wn.NOUN:
                if not (any(s.hypernyms() for s in wn.synsets(name, pos=pos)) and
                        any(s.hyponyms() for s in wn.synsets(name, pos=pos))): continue
            cands.append(name)
    return sorted(set(cands))


# Rebuild the prototype sample with the same seed
print("Building 212-word prototype sample...", flush=True)
glove_set = set()
with open(GLOVE_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= TOP_K_GLOVE: break
        glove_set.add(line.split(" ", 1)[0])
noun_cands = [w for w in collect_candidates(wn.NOUN) if w in glove_set]
adj_cands  = [w for w in collect_candidates(wn.ADJ) if w in glove_set]
random.seed(SEED)
seeds = random.sample(noun_cands, N_SEEDS_NOUN) + random.sample(adj_cands, N_SEEDS_ADJ)
sample = set(seeds)
for s in seeds:
    nbrs = sorted(get_related(s, ("synonyms", "hypernyms", "hyponyms")) & glove_set)
    random.shuffle(nbrs)
    sample.update(nbrs[:NEIGHBORS_PER_SEED])
sample_set = set(sample)

words, vecs = [], []
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        word, *vec = line.rstrip().split(" ")
        if word in sample_set:
            words.append(word); vecs.append(np.array(vec, dtype=np.float32))
kv = KeyedVectors(vector_size=100)
kv.add_vectors(words, np.stack(vecs))
vocab = set(words)
print(f"  loaded {len(words)} vectors", flush=True)

lex_all = {w: sorted(get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab)
           for w in vocab if get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab}
print(f"  WN_all: {len(lex_all)} words with neighbours", flush=True)


def retrofit_A(kv_in, lexicon, n_iter=10, alpha=1.0):
    # Baseline: string lookups in the inner loop, per-word setitem on output
    original = {w: kv_in[w].copy() for w in kv_in.key_to_index}
    new      = {w: kv_in[w].copy() for w in kv_in.key_to_index}
    for it in range(n_iter):
        for w, neighbors in lexicon.items():
            valid = [n for n in neighbors if n in new]
            if not valid: continue
            beta = 1.0 / len(valid)
            numerator = alpha * original[w] + sum(beta * new[n] for n in valid)
            new[w] = numerator / (alpha + beta * len(valid))
    out = KeyedVectors(vector_size=kv_in.vector_size)
    for w in kv_in.key_to_index:
        out[w] = new[w]
    return out


def retrofit_B(kv_in, lexicon, n_iter=10, alpha=1.0):
    # Optimisation 1 only: index arrays in the inner loop, still per-word setitem on output
    vocab_list = list(kv_in.key_to_index.keys())
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    original = np.stack([kv_in[w] for w in vocab_list])
    new = original.copy()
    nbr_idx = {word2idx[w]: np.array([word2idx[n] for n in neighbors if n in word2idx], dtype=np.int32)
               for w, neighbors in lexicon.items() if w in word2idx}
    for it in range(n_iter):
        for wi, idxs in nbr_idx.items():
            if len(idxs) == 0: continue
            beta = 1.0 / len(idxs)
            new[wi] = (alpha * original[wi] + beta * new[idxs].sum(axis=0)) / (alpha + beta * len(idxs))
    out = KeyedVectors(vector_size=kv_in.vector_size)
    for i, w in enumerate(vocab_list):
        out[w] = new[i]
    return out


def retrofit_C(kv_in, lexicon, n_iter=10, alpha=1.0):
    # Optimisation 2 only: string lookups in the inner loop, matrix assembly + add_vectors on output
    original = {w: kv_in[w].copy() for w in kv_in.key_to_index}
    new      = {w: kv_in[w].copy() for w in kv_in.key_to_index}
    for it in range(n_iter):
        for w, neighbors in lexicon.items():
            valid = [n for n in neighbors if n in new]
            if not valid: continue
            beta = 1.0 / len(valid)
            numerator = alpha * original[w] + sum(beta * new[n] for n in valid)
            new[w] = numerator / (alpha + beta * len(valid))
    vocab_list = list(kv_in.key_to_index.keys())
    matrix = np.stack([new[w] for w in vocab_list])
    out = KeyedVectors(vector_size=kv_in.vector_size)
    out.add_vectors(vocab_list, matrix)
    return out


def retrofit_D(kv_in, lexicon, n_iter=10, alpha=1.0):
    # Both optimisations: index arrays + matrix assembly + add_vectors
    vocab_list = list(kv_in.key_to_index.keys())
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    original = np.stack([kv_in[w] for w in vocab_list])
    new = original.copy()
    nbr_idx = {word2idx[w]: np.array([word2idx[n] for n in neighbors if n in word2idx], dtype=np.int32)
               for w, neighbors in lexicon.items() if w in word2idx}
    for it in range(n_iter):
        for wi, idxs in nbr_idx.items():
            if len(idxs) == 0: continue
            beta = 1.0 / len(idxs)
            new[wi] = (alpha * original[wi] + beta * new[idxs].sum(axis=0)) / (alpha + beta * len(idxs))
    out = KeyedVectors(vector_size=kv_in.vector_size)
    out.add_vectors(vocab_list, new)
    return out


variants = [("A. baseline",          retrofit_A),
            ("B. + index arrays",    retrofit_B),
            ("C. + matrix assembly", retrofit_C),
            ("D. both",              retrofit_D)]

# Warm-up to remove the cost of imports / first calls
print(f"\nWarm-up...", flush=True)
for name, fn in variants:
    fn(kv, lex_all, n_iter=N_ITER)
print("  done", flush=True)

print(f"\nRunning {N_RUNS} timed runs per variant...", flush=True)
results = {}
for name, fn in variants:
    times = []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        out = fn(kv, lex_all, n_iter=N_ITER)
        times.append(time.perf_counter() - t0)
    median = np.median(times)
    results[name] = (out, times, median, min(times), max(times))
    print(f"  {name:22s}  median={median*1000:6.1f} ms  range={min(times)*1000:.1f}-{max(times)*1000:.1f}",
          flush=True)

# Sanity check: all variants should produce numerically equivalent output
print(f"\nChecking that all variants produce the same vectors...", flush=True)
ref = results["A. baseline"][0]
for name, (out, *_) in results.items():
    if name == "A. baseline": continue
    diff = max(np.linalg.norm(out[w] - ref[w]) for w in words)
    status = "OK" if diff < 1e-4 else f"DIFFER ({diff:.2e})"
    print(f"  {name:22s}  max L2 diff: {diff:.2e}  [{status}]", flush=True)

import pandas as pd
rows = [{"variant": name,
         "median_ms": round(results[name][2] * 1000, 2),
         "min_ms":    round(results[name][3] * 1000, 2),
         "max_ms":    round(results[name][4] * 1000, 2),
         "speedup":   round(results["A. baseline"][2] / results[name][2], 2)}
        for name, _ in variants]
df = pd.DataFrame(rows)
print(f"\nFinal results:\n{df.to_string(index=False)}", flush=True)
df.to_csv("results/optimization_benchmark.csv", index=False)
print(f"\nSaved to results/optimization_benchmark.csv", flush=True)
