"""Compare OOV strategies: intersection vs filtering vs mean_synonyms.

For each strategy: run retrofit, evaluate on all three benchmarks, record coverage and time.
fastText is handled separately (requires Kaggle / additional model download).
"""
import sys, os, time, gc
import pandas as pd
sys.path.insert(0, "src")
from preprocessing import load_glove, build_wordnet_lexicon
from retrofit import retrofit
from eval import evaluate_all

GLOVE_PATH = "models/glove.6B.300d.txt"
LEXICON_CACHE = "models/wn_all_lexicon.pkl"
STRATEGIES = ["intersection", "filtering", "mean_synonyms"]

# Load data once
print("Loading data...")
glove = load_glove(GLOVE_PATH)
lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                 cache_path=LEXICON_CACHE)

# Baseline
print("\nBaseline GloVe (no retrofit):")
baseline = evaluate_all(glove)
print(baseline.to_string(index=False))
baseline_rho = dict(zip(baseline["benchmark"], baseline["spearman_rho"]))

# Run each strategy
rows = []
for strat in STRATEGIES:
    print(f"\n{'=' * 60}\nSTRATEGY: {strat}\n{'=' * 60}")
    t0 = time.time()
    retrofitted = retrofit(glove, lexicon, n_iter=10, alpha=1.0, beta="inv_degree",
                            oov_strategy=strat, verbose=True)
    elapsed = time.time() - t0

    results = evaluate_all(retrofitted)
    row = {"strategy": strat, "vocab_size": len(retrofitted.key_to_index),
           "time_sec": round(elapsed, 1)}
    for r in results.itertuples():
        row[f"{r.benchmark}_rho"] = round(r.spearman_rho, 4)
        row[f"{r.benchmark}_delta"] = round(r.spearman_rho - baseline_rho[r.benchmark], 4)
    rows.append(row)

    del retrofitted; gc.collect()

# Summary table
print("\n" + "=" * 80); print("OOV STRATEGY COMPARISON"); print("=" * 80)
df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Best strategy per metric
print("\nBest strategy per benchmark (highest Δ):")
for b in ["rg65", "simlex999", "wordsim353"]:
    col = f"{b}_delta"
    best = df.loc[df[col].idxmax()]
    print(f"  {b:12s}: {best['strategy']:15s} (Δ = {best[col]:+.4f})")

# Save for the report
os.makedirs("results", exist_ok=True)
df.to_csv("results/oov_comparison.csv", index=False)
print(f"\nSaved comparison to results/oov_comparison.csv")
