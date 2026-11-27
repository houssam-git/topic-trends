"""Métriques communes LDA / BERTopic."""
from gensim.models.coherencemodel import CoherenceModel

def coherence_cv(topics, texts, dictionary, topn=10):
    """c_v sur les top-n mots ; on retire les mots hors dictionnaire (utile pour BERTopic)."""
    vocab = set(dictionary.token2id)
    topics = [[w for w in t if w in vocab][:topn] for t in topics]
    topics = [t for t in topics if len(t) >= 2]
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()

def diversity(topics, topn=25):
    """Proportion de mots uniques parmi les top-n de tous les thèmes (1 = aucun recouvrement)."""
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / len(words)

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def external_scores(y_true, y_pred, ignore_outliers=True):
    """ARI / NMI par rapport aux catégories HuffPost (évaluation seulement, jamais d'entraînement)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if ignore_outliers:
        mask = y_pred != -1
        y_true, y_pred = y_true[mask], y_pred[mask]
    return {"ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred)}