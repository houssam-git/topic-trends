"""Baseline LDA (gensim)."""
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

def build_dictionary(docs, no_below=20, no_above=0.5):
    d = Dictionary(docs)
    d.filter_extremes(no_below=no_below, no_above=no_above)
    return d

def train_lda(corpus, dictionary, k, seed, passes=5, workers=1):
    """workers=1 : Colab gratuit n'a que 2 cœurs CPU."""
    return LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k,
                        random_state=seed, passes=passes, workers=workers, chunksize=2000)

def top_words(lda, topn=25):
    """Liste des top mots de chaque thème."""
    return [[w for w, _ in lda.show_topic(t, topn=topn)] for t in range(lda.num_topics)]