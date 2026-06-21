"""Loaders for embeddings and lexicons."""
import os, pickle
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn


def load_glove(path: str | Path, vector_size: int | None = None) -> KeyedVectors:
    """Load GloVe text-format embeddings into KeyedVectors."""
    path = Path(path)
    print(f"Loading GloVe from {path.name}...")
    words, vecs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
    if vector_size is None:
        vector_size = len(vecs[0])
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(words, np.stack(vecs))
    print(f"  loaded {len(words)} vectors, dim={vector_size}")
    return kv


def load_fasttext(path: str | Path) -> KeyedVectors:
    """Load fastText binary embeddings (.bin) into gensim KeyedVectors."""
    from gensim.models.fasttext import load_facebook_vectors
    path = Path(path)
    print(f"Loading fastText from {path.name}...")
    kv = load_facebook_vectors(str(path))
    print(f"  loaded {len(kv.key_to_index)} vectors, dim={kv.vector_size}")
    return kv


def build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                          cache_path: str | Path | None = None,
                          lowercase: bool = True) -> dict[str, list[str]]:
    """Build the full English WordNet lexicon graph."""
    key = tuple(sorted(relations))
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached lexicon from {cache_path}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("relations") == key:
            print(f"  loaded {len(cached['lexicon'])} entries")
            return cached["lexicon"]
        print("  cache mismatch (different relations) — rebuilding")

    print(f"Building WordNet lexicon (relations: {key})...")
    lexicon = {}
    norm = (lambda s: s.lower()) if lowercase else (lambda s: s)
    for synset in wn.all_synsets():
        out = set()
        if "synonyms" in relations:
            out.update(norm(l.name()) for l in synset.lemmas() if "_" not in l.name())
        if "hypernyms" in relations:
            out.update(norm(l.name()) for hyp in synset.hypernyms()
                       for l in hyp.lemmas() if "_" not in l.name())
        if "hyponyms" in relations:
            out.update(norm(l.name()) for hyp in synset.hyponyms()
                       for l in hyp.lemmas() if "_" not in l.name())
        for lemma in synset.lemmas():
            w = norm(lemma.name())
            if "_" in lemma.name():
                continue
            lexicon.setdefault(w, set()).update(out - {w})
    lexicon = {w: sorted(neighbors) for w, neighbors in lexicon.items() if neighbors}
    print(f"  built {len(lexicon)} entries, avg degree "
          f"{np.mean([len(v) for v in lexicon.values()]):.2f}")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({"relations": key, "lexicon": lexicon}, f)
        print(f"  cached to {cache_path}")
    return lexicon


def build_wolf_lexicon(path: str | Path,
                       cache_path: str | Path | None = None) -> dict[str, list[str]]:
    """Parse Wolf (French WordNet) XML and extract synonym relations."""
    import xml.etree.ElementTree as ET
    path = Path(path)

    if cache_path and Path(cache_path).exists():
        print(f"Loading cached Wolf lexicon from {cache_path}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  loaded {len(cached)} entries")
        return cached

    print(f"Parsing Wolf XML from {path.name}...")
    tree = ET.parse(path)
    root = tree.getroot()
    lexicon = {}

    for synset in root.iter("SYNSET"):
        lemmas = []
        synonym = synset.find("SYNONYM")
        if synonym is None:
            continue
        for literal in synonym.iter("LITERAL"):
            if literal.text:
                word = literal.text.strip().lower()
                if word == "_empty_": continue
                if " " in word or "." in word: continue
                if not all(c.isalpha() or c == "-" for c in word):
                    continue
                lemmas.append(word)
        for word in lemmas:
            neighbors = [w for w in lemmas if w != word]
            if neighbors:
                lexicon.setdefault(word, set()).update(neighbors)

    lexicon = {w: sorted(neighbors) for w, neighbors in lexicon.items()}
    print(f"  built {len(lexicon)} entries, avg degree "
          f"{sum(len(v) for v in lexicon.values()) / max(len(lexicon), 1):.2f}")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(lexicon, f)
        print(f"  cached to {cache_path}")
    return lexicon


def build_framenet_lexicon() -> dict[str, list[str]]:
    """Build a lexicon from FrameNet: two words connected if they evoke the same frame."""
    from nltk.corpus import framenet as fn
    print("Building FrameNet lexicon...")
    lexicon = {}
    for frame in fn.frames():
        words = []
        for lu in frame.lexUnit.values():
            word = lu.name.split(".")[0].lower()
            if " " not in word and word.isalpha():
                words.append(word)
        words = list(set(words))
        for word in words:
            neighbors = [w for w in words if w != word]
            if neighbors:
                lexicon.setdefault(word, set()).update(neighbors)
    lexicon = {w: sorted(neighbors) for w, neighbors in lexicon.items()}
    print(f"  built {len(lexicon)} entries")
    print(f"  avg degree: {sum(len(v) for v in lexicon.values()) / max(len(lexicon), 1):.2f}")
    return lexicon


def build_lexicon(name: str, **kwargs) -> dict[str, list[str]]:
    """Unified lexicon loader. Dispatches to the appropriate builder."""
    if name == "wn_syn":
        return build_wordnet_lexicon(relations=("synonyms",), **kwargs)
    elif name == "wn_all":
        return build_wordnet_lexicon(
            relations=("synonyms", "hypernyms", "hyponyms"), **kwargs)
    elif name == "wn_hyper":
        return build_wordnet_lexicon(relations=("hypernyms",), **kwargs)
    elif name == "wn_hypo":
        return build_wordnet_lexicon(relations=("hyponyms",), **kwargs)
    elif name == "framenet":
        return build_framenet_lexicon(**kwargs)
    elif name == "wolf":
        path = kwargs.pop("path", "datasets/wolf-1.0b4.xml")
        return build_wolf_lexicon(path=path, **kwargs)
    else:
        raise ValueError(
            f"Unknown lexicon: '{name}'. "
            f"Choose from: wn_syn, wn_all, wn_hyper, wn_hypo, framenet, wolf"
        )
