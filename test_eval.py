import sys
sys.path.insert(0, 'src')
import numpy as np
from gensim.models import KeyedVectors
from eval import evaluate_all

print('Loading GloVe 100d...')
words, vecs = [], []
with open('models/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip().split()
        words.append(parts[0])
        vecs.append(np.array(parts[1:], dtype=np.float32))

kv = KeyedVectors(vector_size=100)
kv.add_vectors(words, np.stack(vecs))
print(f'Loaded {len(words)} vectors')

print('Running evaluation...')
results = evaluate_all(kv)
print(results.to_string(index=False))