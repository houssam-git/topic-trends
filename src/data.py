"""Chargement, nettoyage, déduplication et statistiques du corpus HuffPost."""
import glob, re
import pandas as pd
from config import DATA_RAW, DATA_PROC, MIN_WORDS

def load_raw() -> pd.DataFrame:
    """Télécharge (si besoin) et charge le JSON lines HuffPost."""
    files = glob.glob(f"{DATA_RAW}/*.json")
    if not files:
        import kagglehub, shutil
        path = kagglehub.dataset_download("rmisra/news-category-dataset")
        for f in glob.glob(f"{path}/*.json"):
            shutil.copy(f, DATA_RAW)
        files = glob.glob(f"{DATA_RAW}/*.json")
    return pd.read_json(files[0], lines=True)

def normalize(s: str) -> str:
    """Normalisation légère, uniquement pour détecter les doublons."""
    return re.sub(r"\W+", " ", s.lower()).strip()

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie le corpus et renvoie un journal des lignes supprimées."""
    log = {"initial": len(df)}

    # RGPD (minimisation) : on ne garde que ce qui sert au projet ; les auteurs sont retirés
    df = df[["date", "category", "headline", "short_description"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    log["date_invalide"] = log["initial"] - len(df)

    # Texte = titre + description (BERTopic garde ce texte quasi brut)
    df["text"] = (df["headline"].fillna("") + ". " + df["short_description"].fillna("")).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # Textes trop courts
    n = len(df)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    log["trop_court"] = n - len(df)

    # Doublons (exacts après normalisation)
    n = len(df)
    df["_norm"] = df["text"].map(normalize)
    df = df.drop_duplicates("_norm").drop(columns="_norm")
    log["doublons"] = n - len(df)

    df = df.sort_values("date").reset_index(drop=True)
    log["final"] = len(df)
    return df, log

def describe(df: pd.DataFrame) -> dict:
    """Statistiques descriptives pour le rapport."""
    words = df["text"].str.split().str.len()
    return {
        "n_docs": len(df),
        "periode": f"{df['date'].min().date()} -> {df['date'].max().date()}",
        "n_categories": df["category"].nunique(),
        "mots_moyenne": round(words.mean(), 1),
        "mots_mediane": int(words.median()),
    }

def save(df: pd.DataFrame, name: str = "corpus_clean.parquet"):
    df.to_parquet(f"{DATA_PROC}/{name}", index=False)