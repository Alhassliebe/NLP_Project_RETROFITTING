"""Convergence analysis on the 212-word prototype sample.

Verifies whether the n_iter=1 optimum observed on full GloVe (Section 5.5)
generalises to a different dataset/graph. Reports Δ cos sim on related pairs
under WN_all for n_iter in {1, 2, 3, 5, 10, 25, 50}.
"""
import random, sys
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn

sys.path.insert(0, "src")
from retrofit import retrofit, cosine_similarity

GLOVE_PATH = "models/glove.6B.100d.txt"
SEED = 42
N_SEEDS_NOUN, N_SEEDS_ADJ = 25, 25
NEIGHBORS_PER_SEED = 4
TOP_K_GLOVE = 20000
N_ITERS = [1, 2, 3, 5, 10, 25, 50]


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


# Rebuild the prototype sample
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
lex_all = {w: sorted(get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab)
           for w in vocab if get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab}
print(f"  {len(words)} words, {len(lex_all)} with neighbours", flush=True)

# All lexicon pairs to evaluate Δ cos sim on
pairs = sorted({tuple(sorted([w, n])) for w, nbrs in lex_all.items() for n in nbrs})
print(f"  {len(pairs)} related pairs for evaluation", flush=True)

# Also draw a control set of unrelated pairs (same logic as Section 5.1)
random.seed(SEED + 1)
control_pairs = set()
all_words = sorted(vocab)
while len(control_pairs) < 30:
    w1, w2 = random.sample(all_words, 2)
    if w2 in get_related(w1, ("synonyms", "hypernyms", "hyponyms")): continue
    control_pairs.add(tuple(sorted([w1, w2])))
control_pairs = sorted(control_pairs)

# Run retrofit at each n_iter and record Δ on related and control pairs
print(f"\nRunning retrofit for n_iter ∈ {N_ITERS}...", flush=True)
print(f"{'n_iter':>7} {'Δ related':>11} {'Δ control':>11}", flush=True)
print("-" * 33, flush=True)
rows = []
for n in N_ITERS:
    out = retrofit(kv, lex_all, n_iter=n)
    related = np.mean([cosine_similarity(out[w1], out[w2]) - cosine_similarity(kv[w1], kv[w2])
                       for w1, w2 in pairs])
    control = np.mean([cosine_similarity(out[w1], out[w2]) - cosine_similarity(kv[w1], kv[w2])
                       for w1, w2 in control_pairs])
    print(f"{n:>7} {related:>+11.4f} {control:>+11.4f}", flush=True)
    rows.append({"n_iter": n, "delta_related": round(float(related), 4),
                 "delta_control": round(float(control), 4)})

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("results/convergence_prototype.csv", index=False)
print(f"\nSaved to results/convergence_prototype.csv", flush=True)
