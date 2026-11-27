"""Prétraitement lourd pour LDA (BERTopic utilisera le texte brut)."""
import spacy
from gensim.models.phrases import Phrases

KEEP_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

def tokenize(texts, batch_size=1000, n_process=2):
    """Lemmatise et filtre : garde les lemmes alphabétiques, non-stopwords, len>2."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    docs = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        docs.append([t.lemma_.lower() for t in doc
                     if t.is_alpha and not t.is_stop and len(t) > 2 and t.pos_ in KEEP_POS])
    return docs

def add_bigrams(docs, min_count=20, threshold=10):
    """Fusionne les paires fréquentes (ex. donald + trump -> donald_trump)."""
    bigram = Phrases(docs, min_count=min_count, threshold=threshold).freeze()
    return [bigram[d] for d in docs]