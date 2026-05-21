"""
Shared utilities.

Owner: Person B (with contributions from all)
"""
from __future__ import annotations
import logging
import numpy as np


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )
    return logging.getLogger("retrofitting")


def vocab_overlap(
    embedding_vocab: set[str],
    lexicon_vocab: set[str],
) -> dict[str, int]:
    """Compute vocabulary overlap statistics between embeddings and a lexicon."""
    intersection = embedding_vocab & lexicon_vocab
    return {
        "embedding_size": len(embedding_vocab),
        "lexicon_size": len(lexicon_vocab),
        "intersection_size": len(intersection),
        "coverage_emb": len(intersection) / max(len(embedding_vocab), 1),
        "coverage_lex": len(intersection) / max(len(lexicon_vocab), 1),
    }
