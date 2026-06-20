"""Qualitative analysis: which word pairs change most under retrofitting?

Owner: Sharon — Person C.

Top-15 pairs with largest INCREASE in cosine similarity (true synonyms moving closer)
and top-15 with largest DECREASE. Analogous to Faruqui et al. (2015) Table 6.
"""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, "src")
from preprocessing import load_glove, build_wordnet_lexicon
from gensim.models import KeyedVectors

GLOVE_PATH = "models/glove.6B.300d.txt"
RETROFITTED_PATH = "models/glove_300d_retrofitted_wn_all.kv"
LEXICON_CACHE = "models/wn_all_lexicon.pkl"
TOP_K = 15

# load both embedding spaces
print("Loading baseline GloVe 300d...")
glove = load_glove(GLOVE_PATH)
print("Loading retrofitted vectors...")
if not os.path.exists(RETROFITTED_PATH):
    raise FileNotFoundError(
        f"Retrofitted vectors not found at '{RETROFITTED_PATH}'.\n"
        "Run notebooks/05_retrofit_full_vocab.py first to generate them."
    )
retrofitted = KeyedVectors.load(RETROFITTED_PATH)

print("Loading WN_all lexicon...")
lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                cache_path=LEXICON_CACHE)

# build unique (w1, w2) pairs that exist in both embedding spaces
print("\nBuilding lexicon pair list...")
emb_vocab = set(glove.key_to_index)
pairs = set()
for w, neighbors in lexicon.items():
    if w not in emb_vocab: continue
    for n in neighbors:
        if n not in emb_vocab: continue
        pairs.add(tuple(sorted([w, n])))
print(f"  total lexicon pairs (in vocab): {len(pairs):,}")

def cos(v1, v2):
    return v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\nComputing Δ cos sim for {len(pairs):,} pairs...")
rows = []
for w1, w2 in pairs:
    before = cos(glove[w1], glove[w2])
    after  = cos(retrofitted[w1], retrofitted[w2])
    rows.append((w1, w2, float(before), float(after), float(after - before)))

df = pd.DataFrame(rows, columns=["word1", "word2", "cos_before", "cos_after", "delta"])
print(f"  done — mean Δ = {df['delta'].mean():+.4f}, median Δ = {df['delta'].median():+.4f}")
print(f"  pairs improved: {(df['delta'] > 0).sum():,}/{len(df):,} ({(df['delta'] > 0).mean()*100:.1f}%)")

top_up   = df.nlargest(TOP_K,  "delta").reset_index(drop=True)
top_down = df.nsmallest(TOP_K, "delta").reset_index(drop=True)

def fmt_table(sub, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(f"{'word1':18s} {'word2':18s} {'before':>8s} {'after':>8s} {'Δ':>8s}")
    print("-" * 64)
    for r in sub.itertuples():
        print(f"{r.word1:18s} {r.word2:18s} {r.cos_before:8.4f} {r.cos_after:8.4f} {r.delta:+8.4f}")

fmt_table(top_up,   f"TOP {TOP_K} PAIRS WITH LARGEST INCREASE IN COSINE SIMILARITY")
fmt_table(top_down, f"TOP {TOP_K} PAIRS WITH LARGEST DECREASE IN COSINE SIMILARITY")

df.to_csv("results/qualitative_full.csv", index=False)
top_up.to_csv("results/qualitative_top_increase.csv", index=False)
top_down.to_csv("results/qualitative_top_decrease.csv", index=False)
print(f"\nSaved to results/qualitative_full.csv, top_increase.csv, top_decrease.csv")
