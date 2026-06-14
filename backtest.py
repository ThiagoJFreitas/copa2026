"""Backtest da qualidade das previsões (placar exato, top-N, resultado, RPS).

Treina o modelo só com jogos ANTERIORES a um ano de corte e testa nos seguintes,
usando exatamente o `predict_match` que o app usa. Serve para re-tunar parâmetros
e comprovar melhorias.

Rodar:  python backtest.py            (corte padrão 2022, goal_scale 1.30)
        python backtest.py 2020 1.45  (corte 2020, goal_scale 1.45)
"""

from __future__ import annotations

import sys

import data
import model


def ranked_probability_score(p_home, p_draw, p_away, outcome) -> float:
    """RPS para 1×2 (categorias ordenadas). Menor = melhor."""
    obs = {"home": (1, 0, 0), "draw": (0, 1, 0), "away": (0, 0, 1)}[outcome]
    preds = (p_home, p_draw, p_away)
    cum_p = cum_o = 0.0
    total = 0.0
    for k in range(2):  # r-1 = 2 termos
        cum_p += preds[k]
        cum_o += obs[k]
        total += (cum_p - cum_o) ** 2
    return total / 2.0


def run(cutoff_year: int = 2022, goal_scale: float = 1.30, w_elo: float = 0.5) -> dict:
    df = data.load_results()
    m = model.build_model(df, cutoff_date=f"{cutoff_year}-01-01",
                          w_elo=w_elo, goal_scale=goal_scale)
    test = df[df["date"].dt.year >= cutoff_year]

    n = exato = top3 = top5 = res_ok = 0
    rps_sum = 0.0
    for r in test.itertuples(index=False):
        if r.home_team not in m.elo or r.away_team not in m.elo:
            continue
        p = m.predict_match(r.home_team, r.away_team, neutral=bool(r.neutral))
        real = (int(r.home_score), int(r.away_score))
        ao = "home" if real[0] > real[1] else ("draw" if real[0] == real[1] else "away")
        tops = [(i, j) for i, j, _ in p["top_scores"]]
        # top5: recomputa a partir da matriz completa
        mat = m._poisson_matrix(model.to_dataset_name(r.home_team),
                                model.to_dataset_name(r.away_team), bool(r.neutral))
        flat = mat.flatten(); order = flat.argsort()[::-1][:5]
        top5set = {(int(o // mat.shape[0]), int(o % mat.shape[0])) for o in order}

        n += 1
        exato += real == (p["score_a"], p["score_b"])
        top3 += real in set(tops)
        top5 += real in top5set
        res_ok += p["result"] == ao
        rps_sum += ranked_probability_score(p["p_win"], p["p_draw"], p["p_loss"], ao)

    return {
        "jogos": n, "exato_%": 100 * exato / n, "top3_%": 100 * top3 / n,
        "top5_%": 100 * top5 / n, "resultado_%": 100 * res_ok / n, "rps": rps_sum / n,
    }


if __name__ == "__main__":
    cy = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    gs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.30
    res = run(cy, gs)
    print(f"Backtest - treino < {cy}, teste >= {cy}, goal_scale={gs}")
    print(f"  jogos avaliados : {res['jogos']}")
    print(f"  placar exato    : {res['exato_%']:.1f}%")
    print(f"  placar no top-3 : {res['top3_%']:.1f}%")
    print(f"  placar no top-5 : {res['top5_%']:.1f}%")
    print(f"  acerto resultado: {res['resultado_%']:.1f}%")
    print(f"  RPS (1x2)       : {res['rps']:.3f}  (menor = melhor; ~0.19 e bom)")
