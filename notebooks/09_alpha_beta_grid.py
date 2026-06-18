"""Grid search over alpha and beta strategies for retrofitting.

For each combination of alpha ∈ {0.5, 1, 2} and beta ∈ {inv_degree, uniform, inv_sq_degree},
run retrofit on GloVe 300d + WN_all + intersection OOV, measure Spearman delta on three
benchmarks. Subprocess isolation per cell to keep RAM bounded.
"""
import sys, os, json, subprocess, time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ALPHAS = [0.5, 1.0, 2.0]
BETAS = ["inv_degree", "uniform", "inv_sq_degree"]
CHILD_MARKER = "--run-cell"


def run_one_cell(alpha: float, beta: str) -> list[dict]:
    """Inner worker: retrofit GloVe 300d with given (alpha, beta), return eval rows."""
    sys.path.insert(0, "src")
    from preprocessing import load_glove, build_wordnet_lexicon
    from retrofit import retrofit
    from eval import evaluate_all

    print(f"[α={alpha}, β={beta}] loading GloVe 300d...", flush=True)
    glove = load_glove("models/glove.6B.300d.txt", vector_size=300)
    print(f"[α={alpha}, β={beta}] loading lexicon...", flush=True)
    lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                     cache_path="models/wn_all_lexicon.pkl")

    print(f"[α={alpha}, β={beta}] baseline eval...", flush=True)
    baseline = evaluate_all(glove)
    baseline_rho = {r.benchmark: r.spearman_rho for r in baseline.itertuples()}

    print(f"[α={alpha}, β={beta}] retrofitting...", flush=True)
    t0 = time.time()
    retrofitted = retrofit(glove, lexicon, n_iter=10, alpha=alpha, beta=beta,
                            oov_strategy="intersection", verbose=False)
    elapsed = time.time() - t0
    print(f"[α={alpha}, β={beta}] retrofit done in {elapsed:.1f}s", flush=True)

    retro = evaluate_all(retrofitted)
    rows = []
    for r in retro.itertuples():
        rows.append({"alpha": alpha, "beta": beta, "benchmark": r.benchmark,
                     "baseline_rho": round(baseline_rho[r.benchmark], 4),
                     "retrofitted_rho": round(r.spearman_rho, 4),
                     "delta": round(r.spearman_rho - baseline_rho[r.benchmark], 4),
                     "time_sec": round(elapsed, 1)})
    return rows


# Child mode
if len(sys.argv) >= 4 and sys.argv[1] == CHILD_MARKER:
    rows = run_one_cell(float(sys.argv[2]), sys.argv[3])
    print("===RESULT===", flush=True)
    print(json.dumps(rows), flush=True)
    sys.exit(0)

# Parent mode: orchestrate 9 subprocesses
print(f"Grid search: {len(ALPHAS)} alphas × {len(BETAS)} betas = {len(ALPHAS) * len(BETAS)} cells",
      flush=True)
all_rows = []
total_cells = len(ALPHAS) * len(BETAS)
cell_idx = 0
for a in ALPHAS:
    for b in BETAS:
        cell_idx += 1
        print(f"\n{'=' * 60}\nCELL {cell_idx}/{total_cells}: α={a}, β={b}\n{'=' * 60}", flush=True)
        result = subprocess.run([sys.executable, "-u", __file__, CHILD_MARKER, str(a), b],
                                 capture_output=True, text=True)
        if result.returncode != 0:
            print(f"!! Subprocess failed (code {result.returncode}):", flush=True)
            print(f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}", flush=True)
            sys.exit(1)
        out_lines = result.stdout.splitlines()
        print("\n".join(out_lines[:-2]), flush=True)  # progress prints
        rows = json.loads(out_lines[-1])
        all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(f"\n{'=' * 60}\nGRID SEARCH RESULTS\n{'=' * 60}", flush=True)
print(df.to_string(index=False))
df.to_csv("results/alpha_beta_grid.csv", index=False)
print("\nSaved to results/alpha_beta_grid.csv", flush=True)

# Heatmap: one per benchmark, showing Δ over the (alpha, beta) grid
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, bm in zip(axes, ["rg65", "simlex999", "wordsim353"]):
    sub = df[df["benchmark"] == bm]
    pivot = sub.pivot(index="alpha", columns="beta", values="delta")
    pivot = pivot[BETAS].sort_index()  # consistent column order
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(BETAS))); ax.set_xticklabels(BETAS, rotation=20, ha="right")
    ax.set_yticks(range(len(ALPHAS))); ax.set_yticklabels([f"α={a}" for a in ALPHAS])
    ax.set_title(f"{bm} (Δ Spearman)")
    for i, a in enumerate(ALPHAS):
        for j, b in enumerate(BETAS):
            v = pivot.loc[a, b]
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                    color="black" if abs(v) > 0.02 else "gray", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle("Grid search: retrofitting Δ as a function of α and β (GloVe 300d, WN_all)", y=1.02)
plt.tight_layout()
plt.savefig("figures/alpha_beta_heatmap.png", dpi=120, bbox_inches="tight")
print("Saved figures/alpha_beta_heatmap.png", flush=True)

# Best cell per benchmark
print("\nBest (α, β) per benchmark:", flush=True)
for bm in ["rg65", "simlex999", "wordsim353"]:
    sub = df[df["benchmark"] == bm]
    best = sub.loc[sub["delta"].idxmax()]
    print(f"  {bm:12s}: α={best['alpha']}, β={best['beta']:15s} → Δ = {best['delta']:+.4f}",
          flush=True)
