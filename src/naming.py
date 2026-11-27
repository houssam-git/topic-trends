"""Nommage des thèmes par LLM local (Ollama) + documents représentatifs."""
import json
import numpy as np
import requests

PROMPT = """You are an expert news analyst. Below is a topic discovered automatically \
in a corpus of HuffPost news headlines (2012-2022).

Keywords (most important first): {keywords}

Representative documents:
{docs}

Give this topic a short, specific label (2 to 5 words, in English) and a one-sentence description.
Answer ONLY with JSON: {{"label": "...", "description": "..."}}"""

def representative_docs(emb, topics, texts, topic_id, n=5):
    """Docs du thème les plus proches du centroïde (similarité cosinus)."""
    idx = np.where(topics == topic_id)[0]
    E = emb[idx]
    c = E.mean(axis=0)
    sims = (E @ c) / (np.linalg.norm(E, axis=1) * np.linalg.norm(c) + 1e-9)
    return [texts[i] for i in idx[np.argsort(-sims)[:n]]]

def ask_ollama(prompt, model="qwen2.5:7b", seed=42, url="http://localhost:11434/api/chat"):
    """Appel déterministe à Ollama, sortie forcée en JSON."""
    r = requests.post(url, json={
        "model": model, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": seed}}, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])

def label_topic(keywords, docs, model="qwen2.5:7b"):
    docs_txt = "\n".join(f"- {d[:250]}" for d in docs)
    prompt = PROMPT.format(keywords=", ".join(keywords), docs=docs_txt)
    try:
        return ask_ollama(prompt, model)
    except Exception as e:
        return {"label": "ERROR", "description": str(e)[:200]}