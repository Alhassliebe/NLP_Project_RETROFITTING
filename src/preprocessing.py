"""
Loaders for embeddings and lexicons.

Owner: Person B
"""
from __future__ import annotations
from pathlib import Path
from gensim.models import KeyedVectors


def load_embeddings(name: str, path: Path) -> KeyedVectors:
    """
    Load pre-trained embeddings.

    Args:
        name: one of {"glove", "word2vec", "fasttext"}
        path: path to the embedding file

    Returns:
        gensim KeyedVectors object
    """
    raise NotImplementedError("Person B: implement in Week 1")


def build_lexicon(name: str) -> dict[str, list[str]]:
    """
    Build a semantic lexicon graph.

    Args:
        name: one of {"wn_syn", "wn_all", "ppdb", "framenet", "wolf"}

    Returns:
        dict mapping each word to its list of related words
    """
    raise NotImplementedError("Person B: implement in Week 1-2")


def build_wordnet_lexicon(relation_types: list[str]) -> dict[str, list[str]]:
    """
    Build a WordNet lexicon with configurable relation types.

    Args:
        relation_types: subset of {"synonyms", "hypernyms", "hyponyms"}
    """
    raise NotImplementedError


def build_ppdb_lexicon(path: Path) -> dict[str, list[str]]:
    """Parse PPDB XL lexical file into a lexicon dict."""
    raise NotImplementedError


def build_framenet_lexicon() -> dict[str, list[str]]:
    """Connect words that evoke the same FrameNet frame."""
    raise NotImplementedError


def build_wolf_lexicon(path: Path) -> dict[str, list[str]]:
    """Parse Wolf (French WordNet) XML into a lexicon dict."""
    raise NotImplementedError
