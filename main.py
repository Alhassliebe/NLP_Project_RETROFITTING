"""
Run the full retrofit pipeline: load embeddings → retrofit → evaluate.

Usage:
    python main.py --embedding glove --lexicon wn_all --benchmark rg65 simlex999
    python main.py --embedding fasttext --lexicon wolf --benchmark rg65
    python main.py --embedding glove --lexicon wn_all --no-retrofit
"""
import argparse
import sys
import os

from src.utils import setup_logging
from src.preprocessing import load_glove, load_fasttext, build_lexicon
from src.retrofit import retrofit
from src.eval import evaluate_all
from src.config import (
    GLOVE_300D_PATH, FASTTEXT_FR_PATH,
    WOLF_PATH, DEFAULT_N_ITER, DEFAULT_ALPHA
)

EMBEDDING_PATHS = {
    "glove":    str(GLOVE_300D_PATH),
    "fasttext": str(FASTTEXT_FR_PATH),
}

LEXICON_CACHE = {
    "wn_syn":   "models/wn_syn_lexicon.pkl",
    "wn_all":   "models/wn_all_lexicon.pkl",
    "wn_hyper": "models/wn_hyper_lexicon.pkl",
    "wn_hypo":  "models/wn_hypo_lexicon.pkl",
    "wolf":     "models/wolf_lexicon.pkl",
    "framenet": None,
}


def main():
    parser = argparse.ArgumentParser(
        description="Retrofitting Word Vectors to Semantic Lexicons (Faruqui et al. 2015)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --embedding glove --lexicon wn_all
  python main.py --embedding glove --lexicon wn_all --benchmark rg65 simlex999 wordsim353
  python main.py --embedding fasttext --lexicon wolf
  python main.py --embedding glove --lexicon wn_syn --n-iter 25
  python main.py --embedding glove --lexicon wn_all --no-retrofit
        """
    )
    parser.add_argument("--embedding",
        choices=["glove", "fasttext"],
        required=True,
        help="Pre-trained embeddings: glove (English) or fasttext (French)")
    parser.add_argument("--lexicon",
        choices=["wn_syn", "wn_all", "wn_hyper", "wn_hypo", "framenet", "wolf"],
        required=True,
        help="Semantic lexicon for retrofitting")
    parser.add_argument("--benchmark",
        nargs="+",
        default=["rg65", "simlex999", "wordsim353"],
        help="Evaluation benchmarks (default: rg65 simlex999 wordsim353)")
    parser.add_argument("--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=f"Number of retrofit iterations (default: {DEFAULT_N_ITER})")
    parser.add_argument("--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Weight of original vector alpha (default: {DEFAULT_ALPHA})")
    parser.add_argument("--beta",
        choices=["inv_degree", "uniform", "inv_sq_degree"],
        default="inv_degree",
        help="Neighbor weight strategy (default: inv_degree = 1/degree)")
    parser.add_argument("--oov-strategy",
        choices=["intersection", "filtering", "mean_synonyms"],
        default="intersection",
        help="OOV handling strategy (default: intersection)")
    parser.add_argument("--no-retrofit",
        action="store_true",
        help="Skip retrofitting, only evaluate baseline embeddings")

    args = parser.parse_args()
    log = setup_logging()

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # load embeddings
    log.info(f"Loading {args.embedding} embeddings...")
    if args.embedding == "glove":
        embeddings = load_glove(EMBEDDING_PATHS["glove"])
    else:
        embeddings = load_fasttext(EMBEDDING_PATHS["fasttext"])

    # baseline eval
    log.info("Evaluating baseline...")
    baseline = evaluate_all(embeddings, args.benchmark)
    print("\n── BASELINE ──────────────────────────────────────────")
    print(baseline[["benchmark", "n_evaluated", "spearman_rho"]].to_string(index=False))

    if args.no_retrofit:
        print("\nSkipping retrofitting (--no-retrofit flag set).")
        return

    # load lexicon
    log.info(f"Building {args.lexicon} lexicon...")
    lexicon_kwargs = {"cache_path": LEXICON_CACHE[args.lexicon]}
    if args.lexicon == "wolf":
        lexicon_kwargs["path"] = str(WOLF_PATH)

    lexicon = build_lexicon(args.lexicon, **lexicon_kwargs)
    log.info(f"Lexicon loaded: {len(lexicon)} entries")

    # retrofit
    log.info(f"Retrofitting: n_iter={args.n_iter}, alpha={args.alpha}, "
             f"beta={args.beta}, oov={args.oov_strategy}")
    retrofitted, conv = retrofit(
        embeddings, lexicon,
        n_iter=args.n_iter,
        alpha=args.alpha,
        beta=args.beta,
        oov_strategy=args.oov_strategy,
        return_convergence=True,
        verbose=True,
    )
    log.info(f"Convergence: {conv[0]:.2f} → {conv[-1]:.2e}")

    # save
    save_path = f"models/{args.embedding}_300d_retrofitted_{args.lexicon}.kv"
    retrofitted.save(save_path)
    log.info(f"Saved retrofitted vectors to {save_path}")

    # eval
    log.info("Evaluating retrofitted embeddings...")
    retro = evaluate_all(retrofitted, args.benchmark)

    print("\n── RESULTS ───────────────────────────────────────────")
    print(f"{'benchmark':12s} {'baseline ρ':>12s} {'retrofitted ρ':>16s} {'Δ':>8s}")
    print("-" * 52)
    for b, r in zip(baseline.itertuples(), retro.itertuples()):
        delta = r.spearman_rho - b.spearman_rho
        sign = "▲" if delta > 0 else "▼"
        print(f"{b.benchmark:12s} {b.spearman_rho:12.4f} "
              f"{r.spearman_rho:16.4f} {sign}{abs(delta):.4f}")

    print(f"\n  embedding={args.embedding}, lexicon={args.lexicon} ({len(lexicon)} entries)")
    print(f"  n_iter={args.n_iter}, alpha={args.alpha}, beta={args.beta}, oov={args.oov_strategy}")
    print(f"  convergence: {conv[0]:.2f} → {conv[-1]:.2e}")
    print(f"  saved to: {save_path}")


if __name__ == "__main__":
    main()
