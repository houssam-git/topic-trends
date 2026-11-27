"""Stabilité entre exécutions : ARI/NMI (partitions) + Jaccard (mots des thèmes)."""
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def ctfidf_words(X, vocab, labels, topn=10):
    """Top mots de chaque thème par c-TF-IDF (même formule que BERTopic),
    recalculés à partir des affectations, sans réentraîner de modèle."""
    labels = np.asarray(labels)
    topics = sorted(set(labels) - {-1})
    tf = np.vstack([np.asarray(X[labels == t].sum(axis=0)).ravel() for t in topics])
    A = tf.sum() / len(topics)                               # nb moyen de mots par thème
    idf = np.log(1 + A / np.maximum(tf.sum(axis=0), 1))
    score = (tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)) * idf
    return {t: set(vocab[np.argsort(-score[i])[:topn]]) for i, t in enumerate(topics)}

def topic_jaccard(wa, wb):
    """Appariement : meilleur Jaccard de chaque thème de A dans B (et inversement), moyenne."""
    j = lambda a, b: len(a & b) / len(a | b)
    ab = np.mean([max(j(a, b) for b in wb.values()) for a in wa.values()])
    ba = np.mean([max(j(b, a) for a in wa.values()) for b in wb.values()])
    return (ab + ba) / 2

def pairwise(runs, words):
    """runs : clé -> affectations ; words : clé -> mots par thème."""
    rows = []
    for a, b in itertools.combinations(runs, 2):
        la, lb = np.asarray(runs[a]), np.asarray(runs[b])
        both = (la != -1) & (lb != -1)                       # docs classés dans les deux runs
        rows.append({"run_a": a, "run_b": b,
                     "ARI": adjusted_rand_score(la, lb),
                     "ARI_sans_outliers": adjusted_rand_score(la[both], lb[both]),
                     "NMI": normalized_mutual_info_score(la, lb),
                     "jaccard_top10": topic_jaccard(words[a], words[b])})
    return pd.DataFrame(rows)