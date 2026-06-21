"""Full-vocabulary retrofitting on GloVe 300d with WN_all and intersection OOV strategy."""
import sys, time, gc, os
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from preprocessing import load_glove, build_wordnet_lexicon
from retrofit import retrofit
from eval import evaluate_all

GLOVE_PATH = "models/glove.6B.300d.txt"
LEXICON_CACHE = "models/wn_all_lexicon.pkl"
OUTPUT_RETROFITTED = "models/glove_300d_retrofitted_wn_all.kv"

print("=" * 60); print("Load data"); print("=" * 60)
t0 = time.time()
glove = load_glove(GLOVE_PATH)
print(f"  GloVe load time: {time.time() - t0:.1f}s")
print(f"  GloVe matrix: ~{glove.vectors.nbytes / 1e6:.0f} MB")

t0 = time.time()
lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                cache_path=LEXICON_CACHE)
print(f"  Lexicon build time: {time.time() - t0:.1f}s")

print("\n" + "=" * 60); print("Baseline GloVe evaluation"); print("=" * 60)
baseline_results = evaluate_all(glove)
print(baseline_results.to_string(index=False))

print("\n" + "=" * 60); print("Retrofitting (intersection OOV)"); print("=" * 60)
t0 = time.time()
retrofitted, conv = retrofit(glove, lexicon, n_iter=10, alpha=1.0,
                              beta="inv_degree", oov_strategy="intersection",
                              return_convergence=True, verbose=True)
print(f"  Retrofit time: {time.time() - t0:.1f}s")
print(f"  Convergence: {conv[0]:.2f} → {conv[-1]:.2e}")

print("\n" + "=" * 60); print("Retrofitted GloVe evaluation"); print("=" * 60)
retro_results = evaluate_all(retrofitted)
print(retro_results.to_string(index=False))

print("\n" + "=" * 60); print("Side-by-side"); print("=" * 60)
print(f"{'benchmark':12s} {'baseline ρ':>12s} {'retrofitted ρ':>16s} {'Δ':>10s}")
print("-" * 56)
for b, r in zip(baseline_results.itertuples(), retro_results.itertuples()):
    delta = r.spearman_rho - b.spearman_rho
    print(f"{b.benchmark:12s} {b.spearman_rho:12.4f} {r.spearman_rho:16.4f} {delta:+10.4f}")

print(f"\nSaving retrofitted vectors to {OUTPUT_RETROFITTED}...")
retrofitted.save(OUTPUT_RETROFITTED)
print(f"  saved ({os.path.getsize(OUTPUT_RETROFITTED) / 1e6:.0f} MB)")

del glove, retrofitted, lexicon; gc.collect()
print("\nDone.")
