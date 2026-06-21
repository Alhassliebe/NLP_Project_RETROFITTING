"""Hyperparameter sensitivity: how do α and β affect retrofitting quality?

Three experiments, GloVe and lexicon loaded once:
  1. α sweep  — vary α ∈ [0.1, 1.0], β fixed at inv_degree  (10 runs)
  2. β sweep  — vary β ∈ [0.1, 1.0], α fixed at 1.0         (10 runs)
  3. 2D grid  — all (α, β) combinations                      (100 runs)

Outputs:
  results/alpha_sweep.csv, results/beta_sweep.csv, results/grid2d.csv
  figures/alpha_beta_curves.png   (3 line plots, one per benchmark)
  figures/alpha_beta_heatmap.png  (3 heatmaps,   one per benchmark)
"""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from preprocessing import load_glove, build_wordnet_lexicon
from retrofit import retrofit
from eval import evaluate_all

N_STEPS  = 10
PARAM_MIN, PARAM_MAX = 0.1, 1.0

ALPHAS = np.round(np.linspace(PARAM_MIN, PARAM_MAX, N_STEPS), 4)
BETAS  = np.round(np.linspace(PARAM_MIN, PARAM_MAX, N_STEPS), 4)

BETA_DEFAULT  = "inv_degree"
ALPHA_DEFAULT = 1.0

BENCHMARKS = ["rg65", "simlex999", "wordsim353"]

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# load once, reuse everywhere
print("Loading GloVe 300d...")
glove = load_glove("models/glove.6B.300d.txt", vector_size=300)

print("Loading lexicon...")
lexicon = build_wordnet_lexicon(
    relations=("synonyms", "hypernyms", "hyponyms"),
    cache_path="models/wn_all_lexicon.pkl",
)

print("Computing baseline...")
baseline_df  = evaluate_all(glove)
baseline_rho = {r.benchmark: r.spearman_rho for r in baseline_df.itertuples()}
print(baseline_df[["benchmark", "spearman_rho"]].to_string(index=False))


def run_one(alpha, beta) -> list[dict]:
    t0 = time.time()
    retrofitted = retrofit(
        glove, lexicon,
        n_iter=10, alpha=alpha, beta=beta,
        oov_strategy="intersection", verbose=False,
    )
    elapsed = time.time() - t0
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
    deltas = "  ".join(f"{r['benchmark']} Δ={r['delta']:+.4f}" for r in rows)
    print(f"{elapsed:.1f}s  |  {deltas}")
    return rows


# ── experiment 1: alpha sweep ─────────────────────────────────────────────────
print(f"\nAlpha sweep ({N_STEPS} steps, β={BETA_DEFAULT})")
alpha_rows = []
for i, a in enumerate(ALPHAS, 1):
    print(f"  [{i:2d}/{N_STEPS}] α={a:.4f} ...", end=" ", flush=True)
    alpha_rows.extend(run_one(a, BETA_DEFAULT))

df_alpha = pd.DataFrame(alpha_rows)
df_alpha.to_csv("results/alpha_sweep.csv", index=False)
print("Saved results/alpha_sweep.csv")

# ── experiment 2: beta sweep ──────────────────────────────────────────────────
print(f"\nBeta sweep ({N_STEPS} steps, α={ALPHA_DEFAULT})")
beta_rows = []
for i, b in enumerate(BETAS, 1):
    print(f"  [{i:2d}/{N_STEPS}] β={b:.4f} ...", end=" ", flush=True)
    beta_rows.extend(run_one(ALPHA_DEFAULT, b))

df_beta = pd.DataFrame(beta_rows)
df_beta.to_csv("results/beta_sweep.csv", index=False)
print("Saved results/beta_sweep.csv")

# ── experiment 3: 2D grid ─────────────────────────────────────────────────────
total = len(ALPHAS) * len(BETAS)
print(f"\n2D grid ({N_STEPS}×{N_STEPS} = {total} runs)")
grid_rows = []
done = 0
for alpha in ALPHAS:
    for beta in BETAS:
        done += 1
        print(f"  [{done:3d}/{total}] α={alpha:.2f}, β={beta:.2f} ...", end=" ", flush=True)
        grid_rows.extend(run_one(alpha, float(beta)))

df_grid = pd.DataFrame(grid_rows)
df_grid.to_csv("results/grid2d.csv", index=False)
print("Saved results/grid2d.csv")


# ── figure 1: line plots (sweeps) ─────────────────────────────────────────────
titles = {"rg65": "RG-65", "simlex999": "SimLex-999", "wordsim353": "WordSim-353"}
colors = {"alpha": "#1f77b4", "beta": "#d62728"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, bm in zip(axes, BENCHMARKS):
    sub_a = df_alpha[df_alpha["benchmark"] == bm].sort_values("alpha")
    sub_b = df_beta[df_beta["benchmark"] == bm].sort_values("beta")
    ax.plot(sub_a["alpha"], sub_a["delta"].values, "o-", color=colors["alpha"],
            linewidth=2, markersize=5, label="α sweep  (β = inv_degree)")
    ax.plot(sub_b["beta"],  sub_b["delta"].values, "s--", color=colors["beta"],
            linewidth=2, markersize=5, label="β sweep  (α = 1.0)")
    ax.axvline(ALPHA_DEFAULT, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.7, label="paper default  α=1")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(titles[bm], fontsize=13)
    ax.set_xlabel("parameter value", fontsize=11)
    ax.set_xlim(0, PARAM_MAX + 0.05)
    ax.set_ylabel("Δ Spearman ρ vs baseline", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Effect of α and β on retrofitting quality  (GloVe 300d, WN_all)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figures/alpha_beta_curves.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/alpha_beta_curves.png")


# ── figure 2: heatmaps (2D grid) ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, bm in zip(axes, BENCHMARKS):
    pivot = df_grid[df_grid["benchmark"] == bm].pivot(
        index="beta", columns="alpha", values="delta"
    )
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                         cmap="RdYlGn", norm=norm, shading="nearest")
    plt.colorbar(mesh, ax=ax, label="Δ Spearman ρ")

    for i, beta in enumerate(pivot.index):
        for j, alpha in enumerate(pivot.columns):
            ax.text(alpha, beta, f"{pivot.loc[beta, alpha]:+.3f}",
                    ha="center", va="center", fontsize=7)

    best_idx   = np.unravel_index(pivot.values.argmax(), pivot.shape)
    best_beta  = pivot.index[best_idx[0]]
    best_alpha = pivot.columns[best_idx[1]]
    best_delta = pivot.values[best_idx]
    ax.plot(best_alpha, best_beta, "k*", markersize=14,
            label=f"best: α={best_alpha:.2f}, β={best_beta:.2f}\nΔ={best_delta:+.3f}")
    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlabel("α", fontsize=11)
    ax.set_ylabel("β", fontsize=11)
    ax.set_title(titles[bm], fontsize=13)
    ax.set_xticks(ALPHAS)
    ax.set_yticks(BETAS)
    ax.tick_params(labelsize=8)

fig.suptitle("2D grid: Δ Spearman ρ over (α, β)  (GloVe 300d, WN_all)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figures/alpha_beta_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved figures/alpha_beta_heatmap.png")


# ── summary ───────────────────────────────────────────────────────────────────
print(f"\nBest α (β={BETA_DEFAULT}):")
for bm in BENCHMARKS:
    sub  = df_alpha[df_alpha["benchmark"] == bm]
    best = sub.loc[sub["delta"].idxmax()]
    print(f"  {bm:12s}: α={best['alpha']:.4f}  Δ={best['delta']:+.4f}")

print(f"\nBest β (α={ALPHA_DEFAULT}):")
for bm in BENCHMARKS:
    sub  = df_beta[df_beta["benchmark"] == bm]
    best = sub.loc[sub["delta"].idxmax()]
    print(f"  {bm:12s}: β={best['beta']:.4f}  Δ={best['delta']:+.4f}")

print("\nBest (α, β) joint:")
for bm in BENCHMARKS:
    sub  = df_grid[df_grid["benchmark"] == bm]
    best = sub.loc[sub["delta"].idxmax()]
    print(f"  {bm:12s}: α={best['alpha']:.4f}, β={best['beta']:.4f}  Δ={best['delta']:+.4f}")
