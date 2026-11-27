"""Dynamique : proportions des thèmes par période + incertitude."""
import numpy as np
import pandas as pd

def topic_shares(dates, topics, freq="Q"):
    """Renvoie (effectifs, proportions, erreur-type, total par période).
    Lignes = thèmes, colonnes = périodes."""
    per = pd.Series(pd.to_datetime(dates)).dt.to_period(freq)
    counts = pd.crosstab(pd.Series(topics, name="topic"), per.rename("period"))
    N = counts.sum(axis=0)
    shares = counts / N
    se = np.sqrt(shares * (1 - shares) / N)       # erreur-type d'une proportion
    return counts, shares, se, N