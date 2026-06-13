"""Carregamento do histórico de jogos de seleções.

Fonte: martj42/international_results (CSV no GitHub, sem chave de API,
atualizado continuamente, cobrindo de 1872 até o presente).

Colunas: date, home_team, away_team, home_score, away_score,
         tournament, city, country, neutral
"""

from __future__ import annotations

import os
import tempfile
import time

import pandas as pd
import requests

CSV_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# Cache no diretório temporário do SO (sempre gravável). Não usamos a pasta do
# projeto porque o "Controlled Folder Access" do Windows pode bloquear escrita
# de processos (python.exe) em Documentos.
CACHE_PATH = os.path.join(tempfile.gettempdir(), "copadomundo_results_cache.csv")
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 horas


def _cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_PATH):
        return False
    age = time.time() - os.path.getmtime(CACHE_PATH)
    return age < CACHE_MAX_AGE_SECONDS


def download_results(force: bool = False) -> str:
    """Baixa o CSV e grava no cache local. Retorna o caminho do cache.

    Usa o cache se ele existir e for recente (< 24h), a menos que force=True.
    Se o download falhar mas existir um cache antigo, reutiliza o cache.
    """
    if not force and _cache_is_fresh():
        return CACHE_PATH

    try:
        resp = requests.get(CSV_URL, timeout=30)
        resp.raise_for_status()
        with open(CACHE_PATH, "wb") as fh:
            fh.write(resp.content)
    except Exception:
        if os.path.exists(CACHE_PATH):
            # Falhou a rede mas temos um cache (mesmo velho): melhor que nada.
            return CACHE_PATH
        raise
    return CACHE_PATH


def load_results(force_download: bool = False) -> pd.DataFrame:
    """Retorna o histórico de jogos como DataFrame com `date` em datetime."""
    path = download_results(force=force_download)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # `neutral` vem como texto "TRUE"/"FALSE" ou bool, dependendo da versão.
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    d = load_results()
    print(f"{len(d):,} jogos de {d['date'].min().date()} a {d['date'].max().date()}")
    print(d.tail(3).to_string())
