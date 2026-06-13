"""Resultados reais dos jogos da Copa 2026 a partir do dataset martj42.

O próprio dataset martj42 é atualizado continuamente e já inclui os jogos da
Copa conforme acontecem — então ele é a "verdade" contra a qual comparamos as
previsões. Casamos cada fixture (data + par de seleções) com a linha do dataset,
orientando o placar para o mandante do nosso fixture.
"""

from __future__ import annotations

import pandas as pd

from teams import canonical

# Início da Copa: para previsão out-of-sample treinamos só com jogos anteriores.
TOURNAMENT_START = pd.Timestamp("2026-06-11")


def world_cup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto do dataset com os jogos da Copa 2026 já disputados (martj42)."""
    tour = df["tournament"].str.contains("FIFA World Cup", case=False, na=False)
    notqual = ~df["tournament"].str.contains("qualif", case=False, na=False)
    mask = (df["date"].dt.year == 2026) & tour & notqual
    return df[mask].copy()


def _key(a: str, b: str) -> frozenset:
    return frozenset({canonical(a), canonical(b)})


def build_index(df: pd.DataFrame, *sources) -> dict:
    """Índice de resultados por par de seleções (frozenset) -> dados do jogo.

    Base = dataset martj42. Cada fonte extra é uma tupla (nome, lista_de_registros)
    aplicada na ordem dada — as últimas têm precedência (fontes ao vivo vencem).
    """
    index = {}
    for row in world_cup_results(df).itertuples(index=False):
        index[_key(row.home_team, row.away_team)] = {
            "home": canonical(row.home_team),
            "away": canonical(row.away_team),
            "hs": int(row.home_score),
            "as": int(row.away_score),
            "date": row.date.date().isoformat(),
            "source": "martj42",
        }
    for name, records in sources:
        for r in records or []:
            index[_key(r["home"], r["away"])] = {
                "home": canonical(r["home"]),
                "away": canonical(r["away"]),
                "hs": int(r["hs"]),
                "as": int(r["as"]),
                "date": r.get("date", ""),
                "source": name,
            }
    return index


def augment_df(df: pd.DataFrame, api_results: list | None) -> pd.DataFrame:
    """Acrescenta ao histórico de treino os jogos da API que faltam no martj42.

    Assim o modelo (especialmente no modo rolling) aprende com os resultados mais
    recentes da Copa que o dataset martj42 ainda não registrou.
    """
    if not api_results:
        return df
    existing = {_key(r.home_team, r.away_team) for r in world_cup_results(df).itertuples(index=False)}
    rows = []
    for r in api_results:
        if _key(r["home"], r["away"]) in existing:
            continue
        rows.append({
            "date": pd.Timestamp(r["date"]) if r.get("date") else TOURNAMENT_START,
            "home_team": canonical(r["home"]),
            "away_team": canonical(r["away"]),
            "home_score": int(r["hs"]),
            "away_score": int(r["as"]),
            "tournament": "FIFA World Cup",
            "city": "", "country": "", "neutral": True,
        })
    if not rows:
        return df
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def get_actual(index: dict, home: str, away: str):
    """(gols_mandante, gols_visitante) orientado ao mandante do fixture, ou None."""
    entry = index.get(_key(home, away))
    if entry is None:
        return None
    if entry["home"] == canonical(home):
        return entry["hs"], entry["as"]
    return entry["as"], entry["hs"]


def source_summary(index: dict) -> dict:
    """Conta quantos resultados vieram de cada fonte."""
    counts = {}
    for e in index.values():
        counts[e["source"]] = counts.get(e["source"], 0) + 1
    return counts


def outcome(hs: int, as_: int) -> str:
    if hs > as_:
        return "home"
    if hs < as_:
        return "away"
    return "draw"
