"""Prototype of retrofitting (Faruqui et al., 2015)"""
import random
from copy import deepcopy
import numpy as np
from nltk.corpus import wordnet as wn

GLOVE_PATH = "models/glove.6B.100d.txt"
SEED = 42
N_SEEDS_NOUN, N_SEEDS_ADJ = 25, 25  # random seed words per POS
NEIGHBORS_PER_SEED = 4               # up to K WN_all neighbors added per seed
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

print(f"Loading GloVe vocabulary (top {TOP_K_GLOVE})...")
glove_vocab = []
with open(GLOVE_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= TOP_K_GLOVE: break
        glove_vocab.append(line.split(" ", 1)[0])
glove_set = set(glove_vocab)

print("Collecting WordNet candidates...")
noun_cands = [w for w in collect_candidates(wn.NOUN) if w in glove_set]
adj_cands = [w for w in collect_candidates(wn.ADJ) if w in glove_set]
print(f"Candidates passing filters: {len(noun_cands)} nouns, {len(adj_cands)} adjectives")

random.seed(SEED)
seeds = random.sample(noun_cands, N_SEEDS_NOUN) + random.sample(adj_cands, N_SEEDS_ADJ)

# Expand each seed with up to K of its WN_all neighbors that are also in GloVe top-K
sample = set(seeds)
for seed in seeds:
    nbrs = sorted(get_related(seed, ("synonyms", "hypernyms", "hyponyms")) & glove_set)
    random.shuffle(nbrs)
    sample.update(nbrs[:NEIGHBORS_PER_SEED])
sample = sorted(sample)
sample_set = set(sample)
print(f"Sample: {len(seeds)} seeds + neighbors = {len(sample)} words total")

vectors = {}
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        word, *vec = line.rstrip().split(" ")
        if word in sample_set:
            vectors[word] = np.array(vec, dtype=np.float32)
print(f"Loaded {len(vectors)}/{len(sample)} GloVe vectors")

# intersection OOV: only keep words present in both GloVe and WN
vocab = set(vectors)
lex_syn = {w: sorted(get_related(w, ("synonyms",)) & vocab) for w in vocab
           if get_related(w, ("synonyms",)) & vocab}
lex_all = {w: sorted(get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab) for w in vocab
           if get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab}
print(f"\nWN_syn coverage: {len(lex_syn)}/{len(vocab)} words have neighbours")
print(f"WN_all coverage: {len(lex_all)}/{len(vocab)} words have neighbours")
if lex_syn: print(f"WN_syn avg degree: {np.mean([len(v) for v in lex_syn.values()]):.2f}")
if lex_all: print(f"WN_all avg degree: {np.mean([len(v) for v in lex_all.values()]):.2f}")

def cosine(v1, v2): return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def retrofit(vectors, lexicon, n_iter=10, alpha=1.0):
    original, new = deepcopy(vectors), deepcopy(vectors)
    log = []
    for it in range(n_iter):
        change = 0.0
        for w, neighbors in lexicon.items():
            if not neighbors: continue
            beta = 1.0 / len(neighbors)
            numerator = alpha * original[w] + sum(beta * new[n] for n in neighbors if n in new)
            updated = numerator / (alpha + beta * len(neighbors))
            change += np.linalg.norm(updated - new[w])
            new[w] = updated
        log.append(change)
    return new, log

def evaluate(orig, retro, lexicon, label):
    pairs = {tuple(sorted([w, n])) for w, neighbors in lexicon.items() for n in neighbors}
    if not pairs:
        print(f"\n{label}: 0 related pairs — skipping"); return
    deltas = [cosine(retro[w1], retro[w2]) - cosine(orig[w1], orig[w2]) for w1, w2 in pairs]
    print(f"\n{label}: {len(pairs)} related pairs")
    print(f"  mean Δ cos sim:   {np.mean(deltas):+.4f}")
    print(f"  median Δ cos sim: {np.median(deltas):+.4f}")
    print(f"  pairs improved:   {sum(d > 0 for d in deltas)}/{len(deltas)}")
    random.seed(SEED)
    show = random.sample(sorted(pairs), min(8, len(pairs)))
    print(f"  examples:")
    print(f"    {'word1':18s} {'word2':18s} {'before':>8s} {'after':>8s} {'delta':>8s}")
    for w1, w2 in show:
        b, a = cosine(orig[w1], orig[w2]), cosine(retro[w1], retro[w2])
        print(f"    {w1:18s} {w2:18s} {b:8.4f} {a:8.4f} {a - b:+8.4f}")

# Control: random unrelated pairs (should NOT improve)
def control(orig, retro, n=30):
    random.seed(SEED + 1)
    words = sorted(orig.keys())
    pairs = set()
    while len(pairs) < n and len(pairs) < len(words) * (len(words) - 1) // 2:
        w1, w2 = random.sample(words, 2)
        if w2 in get_related(w1, ("synonyms", "hypernyms", "hyponyms")): continue
        pairs.add(tuple(sorted([w1, w2])))
    deltas = [cosine(retro[w1], retro[w2]) - cosine(orig[w1], orig[w2]) for w1, w2 in pairs]
    print(f"\nControl ({len(pairs)} unrelated pairs):")
    print(f"  mean Δ cos sim: {np.mean(deltas):+.4f}  (expected ≈ 0 or slightly negative)")

print("\n" + "=" * 60); print("WN_syn (synonyms only)"); print("=" * 60)
retro_syn, conv_syn = retrofit(vectors, lex_syn)
if conv_syn[0] > 0: print(f"Convergence: {conv_syn[0]:.2f} → {conv_syn[-1]:.2e}")
evaluate(vectors, retro_syn, lex_syn, "WN_syn related pairs")
control(vectors, retro_syn)

print("\n" + "=" * 60); print("WN_all (synonyms + hypernyms + hyponyms)"); print("=" * 60)
retro_all, conv_all = retrofit(vectors, lex_all)
print(f"Convergence: {conv_all[0]:.2f} → {conv_all[-1]:.2e}")
evaluate(vectors, retro_all, lex_all, "WN_all related pairs")
control(vectors, retro_all)
