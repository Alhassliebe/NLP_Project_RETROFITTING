"""French retrofitting experiment: fastText-fr + Wolf → evaluate on RG-65-fr."""
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "src")
from preprocessing import load_fasttext, build_wolf_lexicon
from retrofit import retrofit
from utils import vocab_overlap

print("Loading fastText-fr...")
kv = load_fasttext("models/cc.fr.300.bin")

print("\nLoading Wolf lexicon...")
wolf = build_wolf_lexicon("datasets/wolf-1.0b4.xml",
                           cache_path="models/wolf_lexicon.pkl")

emb_vocab  = set(kv.key_to_index.keys())
lex_vocab  = set(wolf.keys()) | {n for ns in wolf.values() for n in ns}
stats = vocab_overlap(emb_vocab, lex_vocab)
print(f"\nVocabulary overlap:")
print(f"  fastText-fr vocab : {stats['embedding_size']:>10,}")
print(f"  Wolf vocab        : {stats['lexicon_size']:>10,}")
print(f"  Intersection      : {stats['intersection_size']:>10,}")
print(f"  Coverage (emb)    : {stats['coverage_emb']:.2%}")
print(f"  Coverage (lex)    : {stats['coverage_lex']:.2%}")

print("\nLoading RG-65-fr...")
pairs = []
with open("datasets/rg65_french.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            pairs.append((parts[0], parts[1], float(parts[2])))
df = pd.DataFrame(pairs, columns=["word1", "word2", "score"])
print(f"  {len(df)} pairs loaded")

def evaluate_rg65(kv, df, label):
    vocab = set(kv.key_to_index.keys())
    keep  = df[df.word1.isin(vocab) & df.word2.isin(vocab)].copy()
    keep["cos"] = [
        float(kv[r.word1] @ kv[r.word2] /
              (np.linalg.norm(kv[r.word1]) * np.linalg.norm(kv[r.word2])))
        for r in keep.itertuples()
    ]
    rho, p = spearmanr(keep["cos"], keep["score"])
    print(f"  [{label}] n={len(keep)}/{len(df)}  Spearman ρ = {rho:.4f}  (p={p:.2e})")
    return rho

print("\nBaseline (before retrofitting):")
rho_before = evaluate_rg65(kv, df, "baseline")

print("\nRetrofitting (filtering OOV, n_iter=10)...")
kv_retro, conv = retrofit(kv, wolf, n_iter=10, alpha=1.0,
                           beta="inv_degree", oov_strategy="filtering",
                           return_convergence=True, verbose=True)
print(f"  Convergence: {conv[0]:.2f} → {conv[-1]:.2e}")

print("\nAfter retrofitting:")
rho_after = evaluate_rg65(kv_retro, df, "retrofitted")


print("\n" + "="*50)
print("FRENCH EXPERIMENT SUMMARY")
print("="*50)
print(f"  Embedding : fastText-fr (cc.fr.300.bin)")
print(f"  Lexicon   : Wolf (French WordNet)")
print(f"  Benchmark : RG-65-fr ({len(df)} pairs)")
print(f"  Before    : ρ = {rho_before:.4f}")
print(f"  After     : ρ = {rho_after:.4f}")
print(f"  Δ         : {rho_after - rho_before:+.4f}")

print("\n" + "="*60)
print("QUALITATIVE ANALYSIS — pairs from Wolf")
print("="*60)

from retrofit import cosine_similarity
