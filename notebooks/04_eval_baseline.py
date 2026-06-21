"""Sanity check: evaluate baseline GloVe on similarity benchmarks before retrofitting."""
import sys
import numpy as np
from gensim.models import KeyedVectors
sys.path.insert(0, "src")
from eval import evaluate_all

# Load GloVe 100d into KeyedVectors
print("Loading GloVe 100d...")
words, vecs = [], []
with open("models/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        word, *vec = line.rstrip().split(" ")
        words.append(word)
        vecs.append(np.array(vec, dtype=np.float32))
kv = KeyedVectors(vector_size=100)
kv.add_vectors(words, np.stack(vecs))
print(f"Loaded {len(words)} GloVe vectors")

# Evaluate baseline on all three benchmarks
print("\nBaseline GloVe 100d evaluation:")
results = evaluate_all(kv)
print(results.to_string(index=False))
