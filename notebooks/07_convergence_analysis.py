"""Convergence analysis: how does retrofit performance depend on n_iter?"""
import sys, time, gc
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from preprocessing import load_glove, build_wordnet_lexicon
from retrofit import retrofit
from eval import evaluate_all

GLOVE_PATH = "models/glove.6B.300d.txt"
LEXICON_CACHE = "models/wn_all_lexicon.pkl"
N_ITERS_TO_TEST = [1, 2, 3, 5, 10, 25, 50]

# Load data
print("Loading data...")
glove = load_glove(GLOVE_PATH)
lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                 cache_path=LEXICON_CACHE)

# Baseline
baseline_rho = {r.benchmark: r.spearman_rho for r in evaluate_all(glove).itertuples()}
print(f"\nBaseline: {baseline_rho}")

# Run retrofit once with the maximum n_iter, recording the convergence log
# Then we can re-evaluate at any intermediate point by re-running with smaller n_iter.
# (Cheaper than running 7 separate times, since each retrofit is ~3-5 seconds.)
rows = []
for n in N_ITERS_TO_TEST:
    print(f"\n--- n_iter = {n} ---")
    t0 = time.time()
    retrofitted, conv = retrofit(glove, lexicon, n_iter=n, alpha=1.0, beta="inv_degree",
                                  oov_strategy="intersection", return_convergence=True)
    elapsed = time.time() - t0
    results = evaluate_all(retrofitted)
    row = {"n_iter": n, "time_sec": round(elapsed, 1),
           "final_L2_change": round(conv[-1], 6) if conv else 0.0}
    for r in results.itertuples():
        row[f"{r.benchmark}_rho"] = round(r.spearman_rho, 4)
        row[f"{r.benchmark}_delta"] = round(r.spearman_rho - baseline_rho[r.benchmark], 4)
    rows.append(row)
    del retrofitted; gc.collect()

df = pd.DataFrame(rows)
print("\n" + "=" * 80); print("CONVERGENCE ANALYSIS"); print("=" * 80)
print(df.to_string(index=False))
df.to_csv("results/convergence_analysis.csv", index=False)
print("\nSaved to results/convergence_analysis.csv")

# Plot: Spearman vs n_iter for all three benchmarks
fig, ax = plt.subplots(figsize=(8, 5))
for b in ["rg65", "simlex999", "wordsim353"]:
    ax.plot(df["n_iter"], df[f"{b}_rho"], "o-", label=b, linewidth=2, markersize=7)
    ax.axhline(baseline_rho[b], linestyle="--", alpha=0.4, color=ax.lines[-1].get_color())
ax.set_xlabel("Number of retrofit iterations")
ax.set_ylabel("Spearman ρ")
ax.set_title("Retrofitting performance vs n_iter (GloVe 300d, WN_all, intersection OOV)")
ax.set_xscale("log")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=120)
print("Saved figure to figures/convergence.png")
