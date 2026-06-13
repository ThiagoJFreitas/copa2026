"""Modelo de previsão: Elo ponderado + Poisson combinados.

- Elo dá a força de cada seleção e probabilidades Vitória/Empate/Derrota.
- Poisson modela ataque/defesa e gera o placar mais provável.
- As probabilidades finais são uma mistura (blend) dos dois.

O histórico é ponderado por:
- recência: decaimento exponencial por meia-vida (jogos recentes pesam mais);
- importância do torneio: Copa/eliminatórias > amistosos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from teams import to_dataset_name

# ---------------------------------------------------------------------------
# Pesos de importância por torneio (casamento por substring, minúsculo).
# ---------------------------------------------------------------------------
TOURNAMENT_WEIGHTS = [
    ("fifa world cup qualification", 0.80),
    ("fifa world cup", 1.00),
    ("confederations cup", 0.75),
    ("nations league", 0.75),
    ("uefa euro", 0.85),
    ("copa américa", 0.85),
    ("copa america", 0.85),
    ("african cup of nations", 0.80),
    ("afc asian cup", 0.80),
    ("gold cup", 0.75),
    ("oceania nations cup", 0.70),
    ("qualification", 0.70),  # qualquer outra eliminatória continental
    ("friendly", 0.30),
]
DEFAULT_TOURNAMENT_WEIGHT = 0.55

# Constantes de Elo
ELO_BASE = 1500.0
ELO_HFA = 65.0  # vantagem de mando (só em jogos não-neutros)
ELO_K = 40.0

MAX_GOALS = 10  # teto da matriz de placares do Poisson


def _tournament_weight(name: str) -> float:
    low = str(name).lower()
    for key, w in TOURNAMENT_WEIGHTS:
        if key in low:
            return w
    return DEFAULT_TOURNAMENT_WEIGHT


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    """PMF de Poisson sem scipy: lam^k * e^-lam / k!."""
    lam = max(lam, 1e-6)
    return np.exp(-lam) * np.power(lam, k) / np.array([math.factorial(int(i)) for i in k])


@dataclass
class PredictionModel:
    elo: dict          # team -> rating
    attack: dict       # team -> força de ataque relativa
    defense: dict      # team -> força de defesa relativa
    league_avg: float  # média de gols por time por jogo (ponderada)
    w_elo: float       # peso do Elo na mistura (0..1)
    goal_scale: float = 1.0  # multiplicador do nível de gols (calibra realismo)

    def _eff_elo(self, t: str) -> float:
        return self.elo.get(t, ELO_BASE)

    def _eff_attack(self, t: str) -> float:
        return max(0.05, self.attack.get(t, 1.0))

    def _eff_defense(self, t: str) -> float:
        return max(0.05, self.defense.get(t, 1.0))

    def expected_goals(self, a: str, b: str, neutral: bool = True) -> tuple[float, float]:
        """Gols esperados (λ) de cada lado, com o nível de gols calibrado."""
        a, b = to_dataset_name(a), to_dataset_name(b)
        hb = 1.0 if neutral else 1.10
        ab = 1.0 if neutral else 0.95
        lam_a = self.league_avg * self._eff_attack(a) * self._eff_defense(b) * hb * self.goal_scale
        lam_b = self.league_avg * self._eff_attack(b) * self._eff_defense(a) * ab * self.goal_scale
        return lam_a, lam_b

    # ------------------------------------------------------------------
    # Previsão de um confronto
    # ------------------------------------------------------------------
    def _elo_probs(self, a: str, b: str, neutral: bool) -> tuple[float, float, float]:
        ra = self._eff_elo(a)
        rb = self._eff_elo(b)
        hfa = 0.0 if neutral else ELO_HFA
        exp_a = 1.0 / (1.0 + 10 ** ((rb - ra - hfa) / 400.0))
        # Empate mais provável quando as forças são parecidas.
        p_draw = 0.36 * (1.0 - 2.0 * abs(exp_a - 0.5))
        p_draw = max(0.06, p_draw)
        p_win = exp_a - 0.5 * p_draw
        p_loss = (1.0 - exp_a) - 0.5 * p_draw
        p_win = max(p_win, 0.0)
        p_loss = max(p_loss, 0.0)
        total = p_win + p_draw + p_loss
        return p_win / total, p_draw / total, p_loss / total

    def _poisson_matrix(self, a: str, b: str, neutral: bool) -> np.ndarray:
        lam_a, lam_b = self.expected_goals(a, b, neutral)
        ks = np.arange(0, MAX_GOALS + 1)
        pa = _poisson_pmf(ks, lam_a)
        pb = _poisson_pmf(ks, lam_b)
        return np.outer(pa, pb)  # M[i, j] = P(A faz i, B faz j)

    def _poisson_probs(self, matrix: np.ndarray) -> tuple[float, float, float]:
        p_win = np.tril(matrix, -1).sum()   # i > j
        p_draw = np.trace(matrix)           # i == j
        p_loss = np.triu(matrix, 1).sum()   # i < j
        total = p_win + p_draw + p_loss
        return p_win / total, p_draw / total, p_loss / total

    def predict_match(self, team_a: str, team_b: str, neutral: bool = True) -> dict:
        """Prevê um confronto. Nomes podem estar na nomenclatura FIFA."""
        a = to_dataset_name(team_a)
        b = to_dataset_name(team_b)

        elo_w, elo_d, elo_l = self._elo_probs(a, b, neutral)
        matrix = self._poisson_matrix(a, b, neutral)
        poi_w, poi_d, poi_l = self._poisson_probs(matrix)

        p_win = self.w_elo * elo_w + (1 - self.w_elo) * poi_w
        p_draw = self.w_elo * elo_d + (1 - self.w_elo) * poi_d
        p_loss = self.w_elo * elo_l + (1 - self.w_elo) * poi_l

        outcomes = {"home": p_win, "draw": p_draw, "away": p_loss}
        result = max(outcomes, key=outcomes.get)

        # Placar mais provável *coerente com o resultado previsto*: restringe a
        # matriz à região do resultado (mandante/empate/visitante) e pega o pico.
        n = matrix.shape[0]
        rr, cc = np.indices((n, n))
        if result == "home":
            mask = rr > cc
        elif result == "away":
            mask = rr < cc
        else:
            mask = rr == cc
        masked = np.where(mask, matrix, -1.0)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        xg_a, xg_b = self.expected_goals(a, b, neutral)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_win": float(p_win),
            "p_draw": float(p_draw),
            "p_loss": float(p_loss),
            "score_a": int(i),
            "score_b": int(j),
            "xg_a": float(xg_a),
            "xg_b": float(xg_b),
            "result": result,  # 'home' | 'draw' | 'away'
            "elo_a": float(self._eff_elo(a)),
            "elo_b": float(self._eff_elo(b)),
        }

    # ------------------------------------------------------------------
    # Simulação de grupo (Monte Carlo sobre placares de Poisson)
    # ------------------------------------------------------------------
    def simulate_group(self, group_matches, n_sims: int = 5000, seed: int = 42) -> dict:
        """Roda Monte Carlo dos jogos do grupo e estima % de classificação (top 2).

        `group_matches`: lista de (home, away) com nomes (FIFA ou dataset).
        Retorna dict por seleção: pontos esperados, gols, % classificação,
        posição média.
        """
        rng = np.random.default_rng(seed)
        teams = sorted({t for m in group_matches for t in m})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)

        # Pré-calcula lambdas de cada jogo (sempre neutro na Copa).
        lambdas = []
        for home, away in group_matches:
            lam_a, lam_b = self.expected_goals(home, away, neutral=True)
            lambdas.append((home, away, lam_a, lam_b))

        # Vetorizado: tudo em arrays (n_sims, n_times).
        pts = np.zeros((n_sims, n))
        gd = np.zeros((n_sims, n))
        gf = np.zeros((n_sims, n))
        for home, away, lam_a, lam_b in lambdas:
            ih, ia = idx[home], idx[away]
            ga = rng.poisson(lam_a, n_sims)
            gb = rng.poisson(lam_b, n_sims)
            home_win = ga > gb
            away_win = gb > ga
            draw = ga == gb
            pts[:, ih] += 3 * home_win + draw
            pts[:, ia] += 3 * away_win + draw
            gd[:, ih] += ga - gb
            gd[:, ia] += gb - ga
            gf[:, ih] += ga
            gf[:, ia] += gb

        # Chave de classificação: pontos > saldo > gols pró (+ruído p/ desempate).
        keys = pts * 1e6 + gd * 1e3 + gf + rng.random((n_sims, n)) * 1e-3
        ranks = (-keys).argsort(axis=1).argsort(axis=1) + 1  # posição de cada time

        return {
            teams[k]: {
                "exp_points": float(pts[:, k].mean()),
                "qualify_pct": float((ranks[:, k] <= 2).mean()),
                "avg_position": float(ranks[:, k].mean()),
            }
            for k in range(n)
        }


# ---------------------------------------------------------------------------
# Construção do modelo a partir do histórico
# ---------------------------------------------------------------------------
def _prepare(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    df = df[df["date"].dt.year >= cutoff_year].copy()
    df["tw"] = df["tournament"].map(_tournament_weight)
    return df


def compute_elo(df: pd.DataFrame, cutoff_year: int = 1990) -> dict:
    """Calcula o Elo final de cada seleção iterando cronologicamente."""
    df = _prepare(df, cutoff_year)
    elo: dict = {}
    for home, away, hs, as_, neutral, tw in zip(
        df["home_team"], df["away_team"], df["home_score"],
        df["away_score"], df["neutral"], df["tw"],
    ):
        ra = elo.get(home, ELO_BASE)
        rb = elo.get(away, ELO_BASE)
        hfa = 0.0 if neutral else ELO_HFA
        exp_a = 1.0 / (1.0 + 10 ** ((rb - ra - hfa) / 400.0))
        if hs > as_:
            actual = 1.0
        elif hs < as_:
            actual = 0.0
        else:
            actual = 0.5
        gd = abs(hs - as_)
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = (11 + gd) / 8.0
        k = ELO_K * tw * g
        change = k * (actual - exp_a)
        elo[home] = ra + change
        elo[away] = rb - change
    return elo


def compute_strengths(
    df: pd.DataFrame, cutoff_year: int = 1990, half_life_days: float = 1825.0
):
    """Calcula força de ataque/defesa ponderada (recência + torneio) por seleção.

    Retorna (attack, defense, league_avg).
    """
    df = _prepare(df, cutoff_year)
    ref_date = df["date"].max()
    age_days = (ref_date - df["date"]).dt.days.to_numpy()
    recency = 0.5 ** (age_days / half_life_days)
    weight = recency * df["tw"].to_numpy()

    # Observações direcionadas: cada jogo vira 2 linhas (time, adversário, gols).
    home = pd.DataFrame({
        "team": df["home_team"].to_numpy(), "opp": df["away_team"].to_numpy(),
        "gf": df["home_score"].to_numpy(), "ga": df["away_score"].to_numpy(), "w": weight,
    })
    away = pd.DataFrame({
        "team": df["away_team"].to_numpy(), "opp": df["home_team"].to_numpy(),
        "gf": df["away_score"].to_numpy(), "ga": df["home_score"].to_numpy(), "w": weight,
    })
    obs = pd.concat([home, away], ignore_index=True)
    mu = (obs["w"] * obs["gf"]).sum() / obs["w"].sum()  # gols médios por time/jogo

    # Numeradores fixos: gols pró/contra ponderados por time.
    num_a = (obs["w"] * obs["gf"]).groupby(obs["team"]).sum()
    num_d = (obs["w"] * obs["ga"]).groupby(obs["team"]).sum()
    w_by_opp = obs["w"]  # reutilizado nos denominadores

    # Ponto-fixo: ataque/defesa AJUSTADOS pela força do adversário.
    # Esperado de gols do time i vs j = mu * a_i * d_j  ->  resolve a_i, d_i.
    teams = num_a.index
    a = pd.Series(1.0, index=teams)
    d = pd.Series(1.0, index=teams)
    for _ in range(12):
        denom_a = (w_by_opp * obs["opp"].map(d)).groupby(obs["team"]).sum()
        a = (num_a / (mu * denom_a)).clip(0.2, 3.5)
        a = a / a.mean()
        denom_d = (w_by_opp * obs["opp"].map(a)).groupby(obs["team"]).sum()
        d = (num_d / (mu * denom_d)).clip(0.2, 3.5)
        d = d / d.mean()

    return a.fillna(1.0).to_dict(), d.fillna(1.0).to_dict(), float(mu)


def build_model(
    df: pd.DataFrame,
    cutoff_year: int = 1990,
    half_life_days: float = 1825.0,
    w_elo: float = 0.5,
    goal_scale: float = 1.0,
    cutoff_date=None,
) -> PredictionModel:
    """Constrói o modelo.

    cutoff_date: se informado (Timestamp), treina apenas com jogos ANTERIORES a
    essa data — usado para previsão out-of-sample (o modelo não "vê" os
    resultados que vai prever).
    goal_scale: multiplicador do nível de gols (calibra o realismo dos placares).
    """
    if cutoff_date is not None:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]

    elo = compute_elo(df, cutoff_year)
    attack, defense, league_avg = compute_strengths(df, cutoff_year, half_life_days)

    return PredictionModel(
        elo=elo, attack=attack, defense=defense, league_avg=league_avg,
        w_elo=w_elo, goal_scale=goal_scale,
    )


if __name__ == "__main__":
    import data

    d = data.load_results()
    m = build_model(d)
    top = sorted(m.elo.items(), key=lambda x: -x[1])[:10]
    print("Top 10 Elo:")
    for t, r in top:
        print(f"  {t:20s} {r:7.1f}")
    print()
    print("Brazil x Scotland:", m.predict_match("Brazil", "Scotland"))
