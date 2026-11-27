"""Configuration globale : chemins, seeds, paramètres."""
import os, random
import numpy as np

SEED = 42
SEEDS = [42, 7, 123, 2024, 999]      # pour l'étude de stabilité (phase 6)

# Sur Colab : données sur Drive (persistantes). En local : dossier data/
ROOT = "/content/drive/MyDrive/topic-trends" if os.path.exists("/content/drive") else "."
DATA_RAW  = f"{ROOT}/data/raw"
DATA_PROC = f"{ROOT}/data/processed"
OUTPUTS   = f"{ROOT}/outputs"

# Rupture de collecte constatée dans les données (EDA, phase 1)

BREAK_DATE = "2018-06-01"      # avant : ~2 000-2 900 docs/mois ; après : ~150
STABLE_PERIOD = ("2012-01-01", "2018-05-31")   # pour l'analyse de sensibilité

MIN_WORDS = 5                         # textes plus courts = trop peu informatifs

def set_seed(seed: int = SEED):
    """Fixe les sources d'aléatoire (python, numpy, torch si dispo)."""
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def make_dirs():
    for d in [DATA_RAW, DATA_PROC, f"{OUTPUTS}/figures", f"{OUTPUTS}/models", f"{OUTPUTS}/tables"]:
        os.makedirs(d, exist_ok=True)