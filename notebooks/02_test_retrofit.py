"""Sanity check: src/retrofit.py reproduces the prototype's behaviour on the sample."""
import random, sys, numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn

sys.path.insert(0, "src")
from retrofit import retrofit, cosine_similarity

GLOVE_PATH = "models/glove.6B.100d.txt"
SEED = 42
N_SEEDS_NOUN, N_SEEDS_ADJ = 25, 25
NEIGHBORS_PER_SEED = 4
TOP_K_GLOVE = 20000

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
                has_hyper = any(s.hypernyms() for s in wn.synsets(name, pos=pos))
                has_hypo = any(s.hyponyms() for s in wn.synsets(name, pos=pos))
                if not (has_hyper and has_hypo): continue
            cands.append(name)
    return sorted(set(cands))

# same sample as nb01
glove_set = set()
with open(GLOVE_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= TOP_K_GLOVE: break
        glove_set.add(line.split(" ", 1)[0])

noun_cands = [w for w in collect_candidates(wn.NOUN) if w in glove_set]
adj_cands = [w for w in collect_candidates(wn.ADJ) if w in glove_set]
random.seed(SEED)
seeds = random.sample(noun_cands, N_SEEDS_NOUN) + random.sample(adj_cands, N_SEEDS_ADJ)
sample = set(seeds)
for seed in seeds:
    nbrs = sorted(get_related(seed, ("synonyms", "hypernyms", "hyponyms")) & glove_set)
    random.shuffle(nbrs)
    sample.update(nbrs[:NEIGHBORS_PER_SEED])
sample_set = set(sample)

words, vecs = [], []
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        word, *vec = line.rstrip().split(" ")
        if word in sample_set:
            words.append(word)
            vecs.append(np.array(vec, dtype=np.float32))
kv = KeyedVectors(vector_size=100)
kv.add_vectors(words, np.stack(vecs))
print(f"Loaded {len(words)} words into KeyedVectors")

vocab = set(words)
lex_all = {w: sorted(get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab) for w in vocab
           if get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab}

new_kv, conv = retrofit(kv, lex_all, n_iter=10, return_convergence=True)
print(f"\nConvergence: {conv[0]:.4f} → {conv[-1]:.2e}")
assert conv[-1] < conv[0] * 1e-3, "Algorithm did not converge"

pairs = {tuple(sorted([w, n])) for w, neighbors in lex_all.items() for n in neighbors}
deltas = [cosine_similarity(new_kv[w1], new_kv[w2]) - cosine_similarity(kv[w1], kv[w2])
          for w1, w2 in pairs]
print(f"\nRelated pairs: {len(pairs)}")
print(f"  mean Δ cos sim:  {np.mean(deltas):+.4f}")
print(f"  pairs improved:  {sum(d > 0 for d in deltas)}/{len(deltas)}")

print("\n✓ Production retrofit() works correctly.")
