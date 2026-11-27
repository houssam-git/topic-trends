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