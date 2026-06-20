"""Sentiment analysis on SST-2: baseline GloVe vs retrofitted GloVe.

Owner: Sharon — Person C.

Pipeline (Faruqui et al. 2015 Section 5):
average word vectors per sentence -> logistic regression -> accuracy + F1 on SST-2.
"""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, "src")
from preprocessing import load_glove
from eval import evaluate_sentiment

GLOVE_PATH = "models/glove.6B.300d.txt"
RETROFITTED_PATH = "models/glove_300d_retrofitted_wn_all.kv"

# load SST-2 from HuggingFace
from datasets import load_dataset
print("Loading SST-2...")
sst = load_dataset("stanfordnlp/sst2")
print(f"  train: {len(sst['train'])}, validation: {len(sst['validation'])}")

def tokenize(sentence):
    return [t for t in "".join(c if c.isalpha() or c.isspace() else " "
                                for c in sentence.lower()).split() if t]

def to_dataframe(split):
    return pd.DataFrame({"tokens": [tokenize(ex["sentence"]) for ex in split],
                         "label":  [ex["label"] for ex in split]})

print("Tokenizing...")
train_df = to_dataframe(sst["train"])
test_df  = to_dataframe(sst["validation"])
print(f"  train: {len(train_df)} sentences, test: {len(test_df)}")
print(f"  avg tokens/sentence: {train_df['tokens'].apply(len).mean():.1f}")
print(f"  label balance (train): {train_df['label'].mean():.3f}")

print("\n" + "=" * 60); print("BASELINE GloVe 300d"); print("=" * 60)
glove = load_glove(GLOVE_PATH)
baseline = evaluate_sentiment(glove, train_df, test_df)
print(f"  accuracy: {baseline['accuracy']:.4f}")
print(f"  F1:       {baseline['f1']:.4f}")

print("\n" + "=" * 60); print("RETROFITTED GloVe 300d (WN_all, intersection)"); print("=" * 60)
from gensim.models import KeyedVectors
if os.path.exists(RETROFITTED_PATH):
    print(f"Loading retrofitted vectors from {RETROFITTED_PATH}...")
    retrofitted = KeyedVectors.load(RETROFITTED_PATH)
else:
    print("Retrofitted vectors not found — running retrofit from scratch...")
    from preprocessing import build_wordnet_lexicon
    from retrofit import retrofit
    lexicon = build_wordnet_lexicon(relations=("synonyms", "hypernyms", "hyponyms"),
                                    cache_path="models/wn_all_lexicon.pkl")
    retrofitted = retrofit(glove, lexicon, n_iter=10, alpha=1.0, beta="inv_degree",
                           oov_strategy="intersection")

retro = evaluate_sentiment(retrofitted, train_df, test_df)
print(f"  accuracy: {retro['accuracy']:.4f}")
print(f"  F1:       {retro['f1']:.4f}")

print("\n" + "=" * 60); print("COMPARISON"); print("=" * 60)
print(f"{'metric':12s} {'baseline':>12s} {'retrofitted':>14s} {'Δ':>10s}")
print("-" * 52)
for k in ["accuracy", "f1"]:
    delta = retro[k] - baseline[k]
    print(f"{k:12s} {baseline[k]:12.4f} {retro[k]:14.4f} {delta:+10.4f}")

results = pd.DataFrame([
    {"setting": "baseline",            "accuracy": baseline["accuracy"], "f1": baseline["f1"]},
    {"setting": "retrofitted_wn_all",  "accuracy": retro["accuracy"],    "f1": retro["f1"]},
])
results.to_csv("results/sentiment_analysis.csv", index=False)
print(f"\nSaved to results/sentiment_analysis.csv")
