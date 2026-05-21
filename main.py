"""
Entry point: run the full pipeline (load -> retrofit -> evaluate).

Usage:
    python main.py --embedding glove --lexicon wn_all --benchmark rg65 simlex999
"""
import argparse
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Retrofitting pipeline")
    parser.add_argument("--embedding", choices=["glove", "word2vec", "fasttext"], required=True)
    parser.add_argument("--lexicon", choices=["wn_syn", "wn_all", "ppdb", "framenet", "wolf"], required=True)
    parser.add_argument("--benchmark", nargs="+", default=["rg65", "wordsim353_sim", "simlex999"])
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--oov-strategy", choices=["intersection", "filtering", "mean_synonyms", "fasttext"], default="intersection")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"Running pipeline with args: {args}")
    raise NotImplementedError("Wire up the full pipeline in Week 2")


if __name__ == "__main__":
    main()
