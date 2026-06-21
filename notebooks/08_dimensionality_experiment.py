"""Dimensionality experiment: retrofit GloVe at dim ∈ {50, 100, 200, 300} and measure Δ.

Memory-conscious version: each dimension is processed in a subprocess, so memory is
fully released between runs. Run with: python notebooks/08_dimensionality_experiment.py
"""
import sys, os, json, gc, subprocess
import pandas as pd
import matplotlib.pyplot as plt

DIMS = [50, 100, 200, 300]
CHILD_MARKER = "--run-one"


def run_one_dim(d: int) -> list[dict]:
    """Inner worker: load GloVe of one dim, retrofit, eval, return JSON-serialisable rows."""
    sys.path.insert(0, "src")
    from preprocessing import load_glove, build_wordnet_lexicon
    from retrofit import retrofit
    from eval import evaluate_all

    print(f"[dim={d}] loading GloVe...")
    glove = load_glove(f"models/glove.6B.{d}d.txt", vector_size=d)

    print(f"[dim={d}] loading lexicon...")
    lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                     cache_path="models/wn_all_lexicon.pkl")

    print(f"[dim={d}] baseline eval...")
    baseline = evaluate_all(glove)

    print(f"[dim={d}] retrofitting...")
    import time
    t0 = time.time()
    retrofitted = retrofit(glove, lexicon, n_iter=10, alpha=1.0, beta="inv_degree",
                            oov_strategy="intersection", verbose=False)
    elapsed = time.time() - t0
    print(f"[dim={d}] retrofit done in {elapsed:.1f}s")

    retro = evaluate_all(retrofitted)
    rows = []
    for b, r in zip(baseline.itertuples(), retro.itertuples()):
        rows.append({
            "dim": d, "benchmark": b.benchmark,
            "baseline_rho": round(b.spearman_rho, 4),
            "retrofitted_rho": round(r.spearman_rho, 4),
            "delta": round(r.spearman_rho - b.spearman_rho, 4),
            "retrofit_time_sec": round(elapsed, 1),
        })
    return rows


# Child mode: invoked by parent as subprocess, runs one dim, prints JSON, exits
if len(sys.argv) >= 3 and sys.argv[1] == CHILD_MARKER:
    rows = run_one_dim(int(sys.argv[2]))
    print("===RESULT===")
    print(json.dumps(rows))
    sys.exit(0)

# Parent mode: orchestrate one subprocess per dimension
print(f"Dimensionality experiment over dims={DIMS}")
all_rows = []
for d in DIMS:
    print(f"\n{'=' * 60}\nDIMENSION: {d}d (subprocess)\n{'=' * 60}")
    result = subprocess.run([sys.executable, "-u", __file__, CHILD_MARKER, str(d)],
                             capture_output=True, text=True)
    if result.returncode != 0:
        print(f"!! subprocess for dim={d} failed (code {result.returncode})")
        print(f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        sys.exit(1)
    out_lines = result.stdout.splitlines()
    print("\n".join(out_lines[:-2]))
    rows = json.loads(out_lines[-1])
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(f"\n{'=' * 60}\nDIMENSIONALITY EXPERIMENT RESULTS\n{'=' * 60}")
print(df.to_string(index=False))
df.to_csv("results/dimensionality_experiment.csv", index=False)
print("\nSaved to results/dimensionality_experiment.csv")

# Plot: baseline vs retrofitted across dimensions, one panel per benchmark
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, b in zip(axes, ["rg65", "simlex999", "wordsim353"]):
    sub = df[df["benchmark"] == b].sort_values("dim")
    ax.plot(sub["dim"], sub["baseline_rho"], "o--", label="baseline", linewidth=2, markersize=8)
    ax.plot(sub["dim"], sub["retrofitted_rho"], "s-", label="retrofitted", linewidth=2, markersize=8)
    for _, row in sub.iterrows():
        ax.annotate(f"+{row['delta']:.3f}", xy=(row["dim"], row["retrofitted_rho"]),
                    xytext=(5, 8), textcoords="offset points", fontsize=9, color="green")
    ax.set_title(b); ax.set_xlabel("GloVe dim"); ax.set_ylabel("Spearman ρ")
    ax.grid(alpha=0.3); ax.legend()
plt.suptitle("Retrofitting effect across embedding dimensionality (WN_all, intersection)", y=1.02)
plt.tight_layout()
plt.savefig("figures/dimensionality.png", dpi=120, bbox_inches="tight")
print("Saved figures/dimensionality.png")

# Delta plot
fig, ax = plt.subplots(figsize=(8, 5))
for b in ["rg65", "simlex999", "wordsim353"]:
    sub = df[df["benchmark"] == b].sort_values("dim")
    ax.plot(sub["dim"], sub["delta"], "o-", label=b, linewidth=2, markersize=8)
ax.set_xlabel("GloVe dim"); ax.set_ylabel("Δ Spearman ρ")
ax.set_title("Retrofitting benefit vs embedding dimensionality")
ax.grid(alpha=0.3); ax.legend(); ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("figures/dimensionality_delta.png", dpi=120)
print("Saved figures/dimensionality_delta.png")
