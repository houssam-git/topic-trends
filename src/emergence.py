"""Détecteur de thèmes émergents : règle statistique explicite."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def detect(counts, N, k=8, ratio_min=2.0, min_docs=20, alpha=0.05,
           season_ratio=1.5, season_lag=4):
    """counts : thèmes × périodes (effectifs) ; N : total par période.
    Teste chaque période contre les k précédentes. Renvoie un tableau de toutes les
    paires (thème, période), avec la colonne 'emergent'."""
    periods = counts.columns
    out = []
    for j in range(k, len(periods)):
        q, hist = periods[j], periods[j - k:j]
        x_r, n_r = counts[q], N[q]
        x_h, n_h = counts[hist].sum(axis=1), N[hist].sum()
        p_r, p_h = x_r / n_r, x_h / n_h

        # 1) Test de deux proportions (variance poolée), unilatéral "hausse"
        pool = (x_r + x_h) / (n_r + n_h)
        se = np.sqrt(pool * (1 - pool) * (1 / n_r + 1 / n_h)).replace(0, np.nan)
        z = (p_r - p_h) / se
        pval = np.nan_to_num(norm.sf(z), nan=1.0)
        _, p_adj, _, _ = multipletests(pval, alpha=alpha, method="fdr_bh")

        # 2) Taille d'effet (lissage +0.5 pour éviter la division par 0)
        ratio = ((x_r + 0.5) / n_r) / ((x_h + 0.5) / n_h)

        # 4) Anti-saisonnalité : comparaison au même trimestre l'an passé
        q_ly = q - season_lag
        p_ly = counts[q_ly] / N[q_ly] if q_ly in periods else pd.Series(0, index=counts.index)
        season_ok = p_r > season_ratio * p_ly

        out.append(pd.DataFrame({
            "topic": counts.index, "period": q, "docs": x_r.values, "n_period": n_r,
            "p_recent": p_r.values, "p_hist": p_h.values, "ratio": ratio.values,
            "z": z.values, "p_adj": p_adj, "season_ok": season_ok.values}))

    res = pd.concat(out, ignore_index=True)
    res["emergent"] = ((res.p_adj < alpha) & (res.ratio >= ratio_min)
                       & (res.docs >= min_docs) & res.season_ok)
    return res

def recall(res, expected):
    """expected : liste de (topic, période_début, période_fin, nom).
    Un événement est retrouvé si le thème est émergent au moins une fois dans la fenêtre."""
    rows = []
    for t, start, end, label in expected:
        hit = res[(res.topic == t) & res.emergent
                  & (res.period >= pd.Period(start)) & (res.period <= pd.Period(end))]
        rows.append({"événement": label, "topic": t, "détecté": len(hit) > 0,
                     "trimestres": ", ".join(hit.period.astype(str))})
    return pd.DataFrame(rows)