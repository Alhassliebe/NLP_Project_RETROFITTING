"""Continuous grid search over alpha and beta for retrofitting.

Sweeps alpha ∈ [0.1, 2.0] and beta ∈ [0.1, 1.0] as continuous numeric values
(N_STEPS × N_STEPS grid), runs retrofit on GloVe 300d + WN_all + intersection OOV,
and measures Spearman ρ delta on three benchmarks.

Alpha: trust in original embedding vs neighbours (paper default = 1).
Beta: per-neighbour weight (uniform); beta ≈ 0.1-0.3 reproduces inv_degree behaviour.

Each cell runs in a subprocess to keep memory bounded.
"""
import sys, os, json, subprocess, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

N_STEPS   = 6
ALPHA_MIN, ALPHA_MAX = 0.1, 2.0
BETA_MIN,  BETA_MAX  = 0.1, 1.0

ALPHAS = np.round(np.linspace(ALPHA_MIN, ALPHA_MAX, N_STEPS), 4)
BETAS  = np.round(np.linspace(BETA_MIN,  BETA_MAX,  N_STEPS), 4)

BENCHMARKS   = ["rg65", "simlex999", "wordsim353"]
CHILD_MARKER = "--run-cell"


def run_one_cell(alpha: float, beta: float) -> list[dict]:
    """Run retrofit for one (alpha, beta) pair and return eval results."""
    sys.path.insert(0, "src")
    from preprocessing import load_glove, build_wordnet_lexicon
    from retrofit import retrofit
    from eval import evaluate_all

    print(f"[α={alpha:.4f}, β={beta:.4f}] loading GloVe 300d...")
    glove = load_glove("models/glove.6B.300d.txt", vector_size=300)
    print(f"[α={alpha:.4f}, β={beta:.4f}] loading lexicon...")
    lexicon = build_wordnet_lexicon(
        relations=("synonyms", "hypernyms", "hyponyms"),
        cache_path="models/wn_all_lexicon.pkl",
    )

    baseline     = evaluate_all(glove)
    baseline_rho = {r.benchmark: r.spearman_rho for r in baseline.itertuples()}

    print(f"[α={alpha:.4f}, β={beta:.4f}] retrofitting...")
    t0 = time.time()
    retrofitted = retrofit(
        glove, lexicon,
        n_iter=10, alpha=alpha, beta=beta,
        oov_strategy="intersection", verbose=False,
    )
    elapsed = time.time() - t0
    print(f"[α={alpha:.4f}, β={beta:.4f}] done in {elapsed:.1f}s")

    retro = evaluate_all(retrofitted)
    rows = []
    for r in retro.itertuples():
        rows.append({
            "alpha":           alpha,
            "beta":            beta,
            "benchmark":       r.benchmark,
            "baseline_rho":    round(baseline_rho[r.benchmark], 4),
            "retrofitted_rho": round(r.spearman_rho, 4),
            "delta":           round(r.spearman_rho - baseline_rho[r.benchmark], 4),
            "time_sec":        round(elapsed, 1),
        })
    return rows


# child mode: runs one cell and exits
if len(sys.argv) >= 4 and sys.argv[1] == CHILD_MARKER:
    rows = run_one_cell(float(sys.argv[2]), float(sys.argv[3]))
    print("===RESULT===")
    print(json.dumps(rows))
    sys.exit(0)


# parent: spawn N_STEPS² subprocesses
total_cells = N_STEPS * N_STEPS
print(f"Grid search: α ∈ [{ALPHA_MIN}, {ALPHA_MAX}], β ∈ [{BETA_MIN}, {BETA_MAX}], "
      f"{N_STEPS}×{N_STEPS} = {total_cells} cells")

all_rows = []
cell_idx  = 0
for a in ALPHAS:
    for b in BETAS:
        cell_idx += 1
        print(f"\n{'=' * 50}\nCELL {cell_idx}/{total_cells}: α={a:.4f}, β={b:.4f}\n{'=' * 50}")
        result = subprocess.run(
            [sys.executable, "-u", __file__, CHILD_MARKER, str(a), str(b)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"!! subprocess failed (code {result.returncode}):")
            print(f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            sys.exit(1)
        out_lines = result.stdout.splitlines()
        print("\n".join(out_lines[:-2]))
        rows = json.loads(out_lines[-1])
        all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(f"\n{'=' * 50}\nGRID SEARCH RESULTS\n{'=' * 50}")
print(df.to_string(index=False))
df.to_csv("results/alpha_beta_grid.csv", index=False)
print("\nSaved to results/alpha_beta_grid.csv")


# contour plots
alpha_dense = np.linspace(ALPHA_MIN, ALPHA_MAX, 200)
beta_dense  = np.linspace(BETA_MIN,  BETA_MAX,  200)
A_dense, B_dense = np.meshgrid(alpha_dense, beta_dense)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, bm in zip(axes, BENCHMARKS):
    sub    = df[df["benchmark"] == bm]
    pts    = sub[["alpha", "beta"]].values
    deltas = sub["delta"].values

    Z = griddata(pts, deltas, (A_dense, B_dense), method="cubic")

    vmax = max(abs(deltas.min()), abs(deltas.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cf = ax.contourf(A_dense, B_dense, Z, levels=20, cmap="RdYlGn", norm=norm)
    ax.contour(A_dense, B_dense, Z, levels=10, colors="k", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label="Δ Spearman ρ")

    sc = ax.scatter(
        sub["alpha"], sub["beta"],
        c=deltas, cmap="RdYlGn", norm=norm,
        edgecolors="black", linewidths=0.6, s=60, zorder=5,
    )
    for _, row in sub.iterrows():
        ax.annotate(
            f"{row['delta']:+.3f}",
            (row["alpha"], row["beta"]),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7, color="black",
        )

    # mark paper's defaults: α=1, β≈1/degree ≈ 0.2
    ax.axvline(1.0, color="navy",   linestyle="--", linewidth=1.0, alpha=0.7, label="α=1 (paper)")
    ax.axhline(0.2, color="purple", linestyle="--", linewidth=1.0, alpha=0.7, label="β≈1/deg")

    ax.set_xlabel("α (original-vector weight)", fontsize=10)
    ax.set_ylabel("β (uniform neighbour weight)", fontsize=10)
    ax.set_title(f"{bm}  (Δ Spearman ρ)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

plt.suptitle(
    "Continuous grid search: retrofitting Δ as a function of α and β\n"
    "(GloVe 300d, WN_all, intersection OOV, 10 iterations)",
    y=1.02, fontsize=12,
)
plt.tight_layout()
plt.savefig("figures/alpha_beta_contour.png", dpi=150, bbox_inches="tight")
print("Saved figures/alpha_beta_contour.png")


# best (α, β) per benchmark
print("\nBest (α, β) per benchmark:")
for bm in BENCHMARKS:
    sub  = df[df["benchmark"] == bm]
    best = sub.loc[sub["delta"].idxmax()]
    print(f"  {bm:12s}: α={best['alpha']:.4f}, β={best['beta']:.4f}"
          f"  →  Δ = {best['delta']:+.4f}  (ρ = {best['retrofitted_rho']:.4f})")

# alpha slices (mean Δ at each α, averaged over β)
print("\nMean Δ at each α (averaged over β):")
for bm in BENCHMARKS:
    sub   = df[df["benchmark"] == bm]
    means = sub.groupby("alpha")["delta"].mean()
    parts = "  ".join(f"α={a:.2f}→{v:+.4f}" for a, v in means.items())
    print(f"  {bm:12s}: {parts}")
