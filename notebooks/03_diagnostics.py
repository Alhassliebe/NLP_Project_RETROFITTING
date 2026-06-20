"""Diagnostics for the retrofit prototype: lexicon composition and centroid drift."""
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

# reproduce the same sample as the prototype (same seed)
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
vocab = sorted(sample)
vocab_set = set(vocab)
print(f"Sample: {len(vocab)} words")

# lexicon composition — how many edges per relation type
syn_edges, hyper_edges, hypo_edges = 0, 0, 0
for w in vocab:
    syns = get_related(w, ("synonyms",)) & vocab_set
    hypers = get_related(w, ("hypernyms",)) & vocab_set - syns
    hypos = get_related(w, ("hyponyms",)) & vocab_set - syns - hypers
    syn_edges += len(syns)
    hyper_edges += len(hypers)
    hypo_edges += len(hypos)
print(f"\nEdge contributions in WN_all (undirected pairs counted twice):")
print(f"  synonyms:           {syn_edges}")
print(f"  hypernyms (added):  {hyper_edges}")
print(f"  hyponyms (added):   {hypo_edges}")
print(f"  total WN_all edges: {syn_edges + hyper_edges + hypo_edges}")

print(f"\nConcrete examples — 5 random sample words:")
random.seed(SEED + 10)
for w in random.sample(vocab, 5):
    syns = get_related(w, ("synonyms",)) & vocab_set
    added = (get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab_set) - syns
    print(f"  {w}:")
    print(f"    WN_syn neighbors: {sorted(syns) if syns else '(none)'}")
    print(f"    WN_all adds:      {sorted(added) if added else '(none)'}")

# load, retrofit, measure centroid drift
words, vecs = [], []
with open(GLOVE_PATH, encoding="utf-8") as f:
    for line in f:
        word, *vec = line.rstrip().split(" ")
        if word in vocab_set:
            words.append(word); vecs.append(np.array(vec, dtype=np.float32))
kv = KeyedVectors(vector_size=100)
kv.add_vectors(words, np.stack(vecs))
lex_all = {w: sorted(get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab_set)
           for w in vocab_set if get_related(w, ("synonyms", "hypernyms", "hyponyms")) & vocab_set}
new_kv = retrofit(kv, lex_all, n_iter=10)

orig_mat = np.stack([kv[w] for w in words])
new_mat = np.stack([new_kv[w] for w in words])
orig_centroid, new_centroid = orig_mat.mean(axis=0), new_mat.mean(axis=0)
cos_before = np.mean([cosine_similarity(kv[w], orig_centroid) for w in words])
cos_after = np.mean([cosine_similarity(new_kv[w], new_centroid) for w in words])

# Average pairwise cos sim before/after
random.seed(SEED)
idx_pairs = [random.sample(range(len(words)), 2) for _ in range(500)]
pair_before = np.mean([cosine_similarity(orig_mat[i], orig_mat[j]) for i, j in idx_pairs])
pair_after = np.mean([cosine_similarity(new_mat[i], new_mat[j]) for i, j in idx_pairs])

print(f"\nCentroid drift analysis:")
print(f"  Centroid L2 shift:                       {np.linalg.norm(new_centroid - orig_centroid):.4f}")
print(f"  Mean cos(word, centroid) before retrofit: {cos_before:+.4f}")
print(f"  Mean cos(word, centroid) after retrofit:  {cos_after:+.4f}")
print(f"  Δ:                                       {cos_after - cos_before:+.4f}")
print(f"\nAverage pairwise cosine similarity in the sample:")
print(f"  before retrofit: {pair_before:+.4f}")
print(f"  after retrofit:  {pair_after:+.4f}")
print(f"  Δ:               {pair_after - pair_before:+.4f}")
