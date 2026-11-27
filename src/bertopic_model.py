"""BERTopic : embeddings (cache), construction du modèle, extraction des thèmes."""
import os
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

EMB_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}

def get_embeddings(texts, name, cache_dir, batch_size=256):
    """Calcule les embeddings une seule fois puis les recharge depuis Drive."""
    path = f"{cache_dir}/emb_{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODELS[name], device="cuda")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True).astype(np.float32)
    np.save(path, emb)
    return emb

def build_bertopic(seed, n_neighbors=15, n_components=5, min_cluster_size=100,
                   min_samples=None, min_df=10):
    """Assemble UMAP + HDBSCAN + c-TF-IDF avec des paramètres explicites."""
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                      min_dist=0.0, metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric="euclidean", cluster_selection_method="eom",
                            prediction_data=True)
    # token_pattern \S+ : garde les tokens LDA tels quels (ex. donald_trump)
    vectorizer = CountVectorizer(token_pattern=r"\S+", min_df=min_df)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, top_n_words=25,
                    calculate_probabilities=False, verbose=False)

def topic_words(model, topn=25):
    """Top mots de chaque thème, sans le thème outlier -1."""
    return [[w for w, _ in model.get_topic(t)][:topn]
            for t in sorted(model.get_topics()) if t != -1]