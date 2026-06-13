"""Resultados ao vivo da Copa 2026 via API pública (sem chave) da ESPN.

A API gratuita do API-Football não dá acesso à temporada 2026 no plano free,
então usamos o scoreboard da ESPN como fonte ao vivo dos resultados:

    https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD

Uma requisição por data; sem chave; campos `status.type.completed` e
`competitors[].score` / `competitors[].team.displayName`.
"""

from __future__ import annotations

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"


def fetch_results(dates: list[str]) -> list[dict]:
    """Resultados ENCERRADOS para as datas ISO informadas (ex.: '2026-06-12').

    Retorna lista de {date, home, away, hs, as}. Datas sem jogos ou ainda em
    andamento são simplesmente ignoradas.
    """
    out = []
    for ds in dates:
        try:
            resp = requests.get(SCOREBOARD, params={"dates": ds.replace("-", "")}, timeout=20)
            resp.raise_for_status()
            js = resp.json()
        except Exception:  # noqa: BLE001 — rede/JSON: ignora esta data
            continue
        for ev in js.get("events", []):
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            cs = comp.get("competitors") or []
            try:
                home = next(c for c in cs if c.get("homeAway") == "home")
                away = next(c for c in cs if c.get("homeAway") == "away")
                out.append({
                    "date": ds,
                    "home": home["team"]["displayName"],
                    "away": away["team"]["displayName"],
                    "hs": int(home["score"]),
                    "as": int(away["score"]),
                })
            except (StopIteration, KeyError, ValueError, TypeError):
                continue
    return out
