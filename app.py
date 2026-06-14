"""App Streamlit — previsão da fase de grupos da Copa do Mundo 2026.

Modelo: Elo ponderado + Poisson (com nível de gols calibrável). Previsões são
out-of-sample — o modelo é treinado apenas com jogos ANTERIORES ao início da
Copa, então dá para comparar de forma justa a previsão com o que aconteceu.
Vários presets de modelo + um modo Personalizado.

Rodar:  streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import actuals
import data
import display
import espn
import fixtures
import model
from teams import to_dataset_name

st.set_page_config(page_title="Copa 2026 — Previsão da Fase de Grupos", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Presets de modelo — cada um é uma combinação coerente de parâmetros.
# (w_elo, half_life anos, ano de corte, goal_scale)
# ---------------------------------------------------------------------------
PRESETS = {
    "⚖️ Equilibrado": {
        "w_elo": 0.5, "half_life": 5.0, "cutoff": 1994, "goal_scale": 1.30,
        "desc": "Mistura 50/50 de Elo e Poisson, ~5 anos de memória. Bom padrão geral.",
    },
    "🔥 Forma recente": {
        "w_elo": 0.4, "half_life": 2.0, "cutoff": 2014, "goal_scale": 1.30,
        "desc": "Memória curta (2 anos) e só jogos desde 2014: prioriza o momento atual.",
    },
    "📜 Histórico (Elo puro)": {
        "w_elo": 1.0, "half_life": 8.0, "cutoff": 1990, "goal_scale": 1.4,
        "desc": "Só Elo, memória longa. Favorece potências tradicionais e estabilidade.",
    },
    "⚽ Ofensivo (mais gols)": {
        "w_elo": 0.35, "half_life": 4.0, "cutoff": 2006, "goal_scale": 1.75,
        "desc": "Peso no Poisson e nível de gols alto: placares mais elásticos (3-0, 4-1).",
    },
    "🛡️ Defensivo (poucos gols)": {
        "w_elo": 0.6, "half_life": 6.0, "cutoff": 1998, "goal_scale": 1.10,
        "desc": "Nível de gols baixo e mais Elo: jogos truncados, muitos 1-0 e 2-0.",
    },
    "🛠️ Personalizado": None,  # mostra os sliders
}
DEFAULT_PRESET = "⚖️ Equilibrado"


# ---------------------------------------------------------------------------
# Cache de dados / modelo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Baixando histórico de jogos...")
def get_data(refresh: int = 0) -> pd.DataFrame:
    return data.load_results(force_download=bool(refresh))


@st.cache_data(show_spinner="Calculando Elo e forças das seleções...")
def get_model(refresh: int, cutoff_year: int, half_life_years: float, w_elo: float,
              goal_scale: float, cutoff_iso: str, api_items: tuple = ()):
    """Modelo treinado só com jogos ANTERIORES a `cutoff_iso` (out-of-sample).

    `api_items` injeta resultados recentes (ESPN) no histórico de treino.
    """
    df = get_data(refresh)
    api_results = [dict(zip(("date", "home", "away", "hs", "as"), it)) for it in api_items]
    df = actuals.augment_df(df, api_results)
    return model.build_model(
        df,
        cutoff_year=cutoff_year,
        half_life_days=half_life_years * 365.0,
        w_elo=w_elo,
        goal_scale=goal_scale,
        cutoff_date=pd.Timestamp(cutoff_iso),
    )


@st.cache_data(show_spinner="Simulando os grupos...")
def get_group_sims(refresh: int, cutoff_year: int, half_life_years: float, w_elo: float,
                   goal_scale: float, n_sims: int, cutoff_iso: str,
                   api_items: tuple = ()) -> dict:
    m = get_model(refresh, cutoff_year, half_life_years, w_elo, goal_scale,
                  cutoff_iso, api_items)
    return {
        g: m.simulate_group([(h, a) for (_, h, a, _) in fixtures.matches_for_group(g)],
                            n_sims=n_sims)
        for g in fixtures.GROUPS
    }


@st.cache_data(show_spinner="Buscando resultados ao vivo (ESPN)...")
def get_espn_results(refresh: int, dates: tuple) -> list:
    """Resultados ao vivo da Copa via ESPN (sem chave). Cacheado até o 🔄."""
    return espn.fetch_results(list(dates))


# ---------------------------------------------------------------------------
# Componente visual: barra empilhada V / E / D
# ---------------------------------------------------------------------------
def wdl_bar(p_win: float, p_draw: float, p_loss: float, highlight: str | None = None) -> str:
    w, dr, l = round(p_win * 100), round(p_draw * 100), round(p_loss * 100)
    def seg(width, color, label, key):
        border = "outline:2px solid #ffd54f;outline-offset:-2px;" if highlight == key else ""
        return (f"<div style='width:{width}%;background:{color};{border}'>"
                f"{label if width > 7 else ''}</div>")
    return (
        "<div style='display:flex;height:22px;border-radius:5px;overflow:hidden;"
        "font-size:12px;font-weight:600;color:#fff;text-align:center;line-height:22px'>"
        + seg(w, "#2e7d32", w, "home")
        + seg(dr, "#9e9e9e", dr, "draw")
        + seg(l, "#1565c0", l, "away")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.session_state.setdefault("refresh", 0)

with st.sidebar:
    st.header("⚙️ Modelo")
    preset_name = st.selectbox("Tipo de modelo", list(PRESETS), index=list(PRESETS).index(DEFAULT_PRESET))
    preset = PRESETS[preset_name]

    if preset is None:  # Personalizado: expõe todos os parâmetros
        w_elo = st.slider("Peso Elo × Poisson", 0.0, 1.0, 0.5, 0.05,
                          help="0 = só Poisson (placares) · 1 = só Elo (ratings).")
        half_life_years = st.slider("Meia-vida da recência (anos)", 1.0, 12.0, 5.0, 0.5,
                                    help="Menor = jogos recentes dominam o ataque/defesa.")
        cutoff_year = st.slider("Considerar jogos a partir de", 1950, 2020, 1994, 1)
        goal_scale = st.slider("Nível de gols", 0.8, 2.2, 1.30, 0.05,
                               help="Calibra quantos gols sair. ~1.45 ≈ média realista de "
                                    "Copa (2,6/jogo). Maior = placares mais elásticos.")
    else:
        w_elo = preset["w_elo"]
        half_life_years = preset["half_life"]
        cutoff_year = preset["cutoff"]
        goal_scale = preset["goal_scale"]
        st.caption(preset["desc"])
        st.caption(f"Elo×Poisson **{w_elo:.0%}/{1-w_elo:.0%}** · memória **{half_life_years:.0f} anos** "
                   f"· desde **{cutoff_year}** · gols **{goal_scale:.2f}×**")

    st.divider()
    n_sims = st.select_slider("Simulações por grupo (Monte Carlo)",
                              options=[1000, 2000, 5000, 10000, 20000], value=5000)
    rolling = st.toggle(
        "Modo rolling (reincorpora rodadas)", value=False,
        help="Cada jogo é previsto com tudo que aconteceu até a véspera dele "
             "(round 2 já 'sabe' do round 1). Desligado = previsão fixa pré-Copa.",
    )

    st.divider()
    st.subheader("🎟️ Modo bolão")
    bolao = st.toggle("Otimizar palpite para bolão", value=False,
                      help="Escolhe o placar que MAXIMIZA os pontos esperados na regra "
                           "do bolão — não só o mais provável.")
    if bolao:
        st.caption("Regra: **6** placar exato · **4** vencedor + gols de um time · "
                   "**3** vencedor (sem placar) · **1** gols de um time sem o vencedor · "
                   "empate: **6** exato / **3** sem placar exato.")

    st.divider()
    if st.button("🔄 Atualizar resultados / histórico"):
        st.session_state.refresh += 1
        st.cache_data.clear()

refresh = st.session_state.refresh

df = get_data(refresh)
group_dates = tuple(sorted({m[3] for m in fixtures.MATCHES}))
espn_results = get_espn_results(refresh, group_dates)
index = actuals.build_index(df, ("ESPN", espn_results))
# Resultados frescos (já mesclados/canônicos) alimentam o treino do modelo.
api_items = tuple((e["date"], e["home"], e["away"], e["hs"], e["as"])
                  for e in index.values())
START_ISO = actuals.TOURNAMENT_START.isoformat()

# Data do último jogo conhecido (martj42 + API), para o corte do modo rolling.
played_dates = [pd.Timestamp(e["date"]) for e in index.values() if e["date"]]
latest_played = max([df["date"].max()] + played_dates)
LATEST_ISO = (latest_played + pd.Timedelta(days=1)).isoformat()


def model_for(cutoff_iso: str):
    return get_model(refresh, cutoff_year, half_life_years, w_elo, goal_scale,
                     cutoff_iso, api_items)


def predict(home: str, away: str, date: str):
    """Previsão de um jogo. Em modo rolling usa o modelo treinado até a véspera."""
    cutoff = pd.Timestamp(date).isoformat() if rolling else START_ISO
    return model_for(cutoff).predict_match(home, away, neutral=True)


def bolao_pick(home: str, away: str, date: str):
    """Placar que maximiza pontos esperados na regra do bolão. Retorna (i, j)."""
    cutoff = pd.Timestamp(date).isoformat() if rolling else START_ISO
    i, j, _ = model_for(cutoff).best_points_score(home, away, neutral=True)
    return i, j


# Simulações de grupo: em rolling projetam com tudo que já aconteceu.
sims_cutoff = LATEST_ISO if rolling else START_ISO
sims = get_group_sims(refresh, cutoff_year, half_life_years, w_elo, goal_scale,
                      n_sims, sims_cutoff, api_items)
m = model_for(sims_cutoff)  # usado para a coluna de Elo na classificação

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.markdown(display.TABLE_CSS, unsafe_allow_html=True)
st.title("⚽ Copa do Mundo 2026 — Previsão da Fase de Grupos")
st.markdown(
    f"Previsões **out-of-sample** dos 72 jogos da fase de grupos (modelo **{preset_name}**): "
    "treinado só com jogos **anteriores a 11/06/2026**, combinando **Elo** + **Poisson**. "
    "🟩 vitória mandante · ⬜ empate · 🟦 vitória visitante."
)

# ---------------------------------------------------------------------------
# Aba de comparação previsão × realidade + abas por grupo
# ---------------------------------------------------------------------------
tab_names = ["📊 Previsão × Realidade"] + [f"Grupo {g}" for g in fixtures.GROUPS]
tabs = st.tabs(tab_names)

# ----- Tab 0: comparação -----
with tabs[0]:
    st.subheader("Jogos já disputados: previsto × real")
    src = actuals.source_summary(index)
    live = src.get("ESPN", 0) + src.get("API-Football", 0)
    parts = " · ".join(f"**{k}**: {v}" for k, v in sorted(src.items()))
    if live:
        st.caption(f"🟢 Resultados ao vivo — {parts}. Atualize com o 🔄.")
    elif src.get("martj42"):
        st.caption(f"🟡 Resultados via {parts} (martj42 pode estar atrasado alguns dias).")
    rows, n_ok, n_exact, n_top3, goal_err, prob_actual = [], 0, 0, 0, 0.0, 0.0
    bolao_pts = 0  # pontos que o palpite do bolão teria feito nos jogos disputados
    calib = []  # pares (prob prevista, ocorreu?) para o gráfico de calibração
    for g, home, away, date in fixtures.MATCHES:
        act = actuals.get_actual(index, home, away)
        if act is None:
            continue
        pred = predict(home, away, date)
        ao = actuals.outcome(*act)
        po = pred["result"]
        probs = {"home": pred["p_win"], "draw": pred["p_draw"], "away": pred["p_loss"]}
        for k, p in probs.items():
            calib.append({"pred": p, "occ": 1 if ao == k else 0})
        hit = po == ao
        n_ok += hit
        # palpite mostrado = bolão (se ligado) ou placar mais provável
        pick = bolao_pick(home, away, date) if bolao else (pred["score_a"], pred["score_b"])
        exato = pick == act
        n_exact += exato
        top3 = {(i, j) for i, j, _ in pred["top_scores"]}
        in_top3 = act in top3
        n_top3 += in_top3
        bolao_pts += model.bolao_points(pick, act)
        goal_err += abs(pred["score_a"] - act[0]) + abs(pred["score_b"] - act[1])
        prob_actual += probs[ao]
        rows.append({
            "g": g, "home": home, "away": away,
            "prev": f"{pick[0]}–{pick[1]}",
            "top3": " · ".join(f"{i}-{j}" for i, j, _ in pred["top_scores"]),
            "real": f"{act[0]}–{act[1]}",
            "p_real": probs[ao], "ok": hit,
        })

    if not rows:
        st.info("Nenhum jogo da fase de grupos foi disputado ainda (ou o dataset "
                "ainda não os registrou). Clique em **🔄 Atualizar histórico** "
                "conforme a Copa avança.")
    else:
        n = len(rows)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jogos disputados", n)
        c2.metric("Acerto do resultado", f"{n_ok}/{n}", f"{100*n_ok/n:.0f}%")
        c3.metric("Placar exato", f"{n_exact}/{n}")
        c4.metric("Placar no top-3", f"{n_top3}/{n}")
        if bolao:
            c5.metric("🎟️ Pontos no bolão", bolao_pts, f"{bolao_pts/n:.1f}/jogo")
        else:
            c5.metric("Erro médio de gols", f"{goal_err/n:.2f}")
        st.caption(
            f"Prob. média atribuída ao resultado que realmente ocorreu: "
            f"{100*prob_actual/n:.1f}% (quanto maior, mais calibrado). "
            "ℹ️ Acertar **placar exato** em futebol tem teto baixo (~14% mesmo para bons "
            "modelos); por isso mostramos os 3 placares mais prováveis — o placar real cai no "
            "**top-3 em ~36%** dos jogos (histórico). Acerto de resultado fica ~59%."
        )
        headers = [("Grp", "pos"), ("Jogo", "nm"), ("Prev", ""), ("Top-3", ""),
                   ("Real", ""), ("✓", ""), ("P(real)", "")]
        trs = []
        for r in rows:
            jogo = (f"{display.img(r['home'])}{display.pt(r['home'])} "
                    f"<span style='color:#888'>×</span> "
                    f"{display.img(r['away'])}{display.pt(r['away'])}")
            cells = [(r["g"], "pos"), (jogo, "nm"), (f"<b>{r['prev']}</b>", ""),
                     (r["top3"], ""), (r["real"], ""),
                     ("✅" if r["ok"] else "❌", ""), (display.qbar(r["p_real"] * 100), "")]
            trs.append((cells, ""))
        st.markdown(display.html_table(headers, trs), unsafe_allow_html=True)

        # --- Gráfico de calibração (reliability diagram) ---
        st.markdown("#### Calibração do modelo")
        st.caption(
            "Cada ponto agrupa previsões com probabilidade parecida (V/E/D de todos os "
            "jogos disputados). No eixo X, a probabilidade média prevista; no Y, a frequência "
            "real com que aconteceu. Quanto mais perto da linha tracejada (ideal), mais "
            "calibrado. Tende a ficar útil conforme mais jogos são disputados."
        )
        cdf = pd.DataFrame(calib)
        cdf["bin"] = (cdf["pred"] * 5).round() / 5  # bins de 0,2
        agg = (cdf.groupby("bin")
               .agg(pred_mean=("pred", "mean"), obs_freq=("occ", "mean"), n=("occ", "size"))
               .reset_index())
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[5, 5], color="#888").encode(x="x", y="y")
        pts = alt.Chart(agg).mark_circle(opacity=0.85, color="#2e7d32").encode(
            x=alt.X("pred_mean", title="Probabilidade prevista", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("obs_freq", title="Frequência real", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("n", title="nº previsões"),
            tooltip=["pred_mean", "obs_freq", "n"],
        )
        st.altair_chart((diag + pts).properties(height=320), use_container_width=True)

# ----- Tabs por grupo -----
for tab, g in zip(tabs[1:], fixtures.GROUPS):
    with tab:
        col_table, col_matches = st.columns([1, 1.4])

        with col_table:
            st.subheader(f"Classificação projetada — Grupo {g}")
            proj = sorted(sims[g].items(),
                          key=lambda kv: (-kv[1]["qualify_pct"], -kv[1]["exp_points"]))
            headers = [("#", "pos"), ("Seleção", "nm"), ("Elo", ""), ("Pts", ""),
                       ("Classif.", "")]
            trs = []
            for rank, (team, s) in enumerate(proj, 1):
                ds = to_dataset_name(team)
                cells = [(str(rank), "pos"), (display.label_html(team), "nm"),
                         (str(round(m.elo.get(ds, model.ELO_BASE))), ""),
                         (f"{s['exp_points']:.1f}", ""),
                         (display.qbar(s["qualify_pct"] * 100), "")]
                trs.append((cells, "q" if rank <= 2 else ""))
            st.markdown(display.html_table(headers, trs), unsafe_allow_html=True)
            st.caption("Top 2 (verde) avançam · Pts = pontos esperados · "
                       "Classif. = prob. de classificação (Monte Carlo).")

            # --- Classificação REAL (jogos já disputados) ---
            st.markdown(f"##### Classificação atual — Grupo {g}")
            gm = [(h, a) for (_, h, a, _) in fixtures.matches_for_group(g)]
            rstats = actuals.real_standings(index, gm)
            if sum(s["P"] for s in rstats.values()) == 0:
                st.caption("Nenhum jogo disputado ainda neste grupo.")
            else:
                cur = sorted(rstats.items(),
                             key=lambda kv: (-kv[1]["Pts"], -kv[1]["SG"], -kv[1]["GP"]))
                headers = [("#", "pos"), ("Seleção", "nm"), ("P", ""), ("V", ""),
                           ("E", ""), ("D", ""), ("SG", ""), ("Pts", "")]
                trs = []
                for rank, (team, s) in enumerate(cur, 1):
                    cells = [(str(rank), "pos"), (display.label_html(team), "nm"),
                             (str(s["P"]), ""), (str(s["V"]), ""), (str(s["E"]), ""),
                             (str(s["D"]), ""), (f"{s['SG']:+d}", ""),
                             (f"<b>{s['Pts']}</b>", "")]
                    trs.append((cells, "q" if rank <= 2 else ""))
                st.markdown(display.html_table(headers, trs), unsafe_allow_html=True)
                st.caption("P=jogos · V/E/D · SG=saldo · Pts=pontos · top 2 em verde.")

        with col_matches:
            st.subheader("Jogos")
            for _, home, away, date in fixtures.matches_for_group(g):
                pred = predict(home, away, date)
                act = actuals.get_actual(index, home, away)
                d_fmt = pd.to_datetime(date).strftime("%d/%m")
                if bolao:
                    bi, bj = bolao_pick(home, away, date)
                    marker = "🎟️ "
                else:
                    bi, bj = pred["score_a"], pred["score_b"]
                    marker = ""
                # vencedor implícito no placar do palpite
                wlabel = (display.pt(home) if bi > bj else
                          (display.pt(away) if bi < bj else "Empate"))
                teams = (f"<div class='teams'>{display.img(home)}<b>{display.pt(home)}</b>"
                         f"<span class='vs'>×</span>"
                         f"{display.img(away)}<b>{display.pt(away)}</b>"
                         f"<span style='color:#8a9099;font-size:12px'>&nbsp;· {d_fmt}</span></div>")
                pred_row = (f"<div class='pred'><span class='lbl'>palpite</span>"
                            f"<span class='sc'>{marker}{bi} × {bj}</span>"
                            f"<span class='pwin'>{wlabel}</span></div>")
                real = ""
                if act is not None:
                    ok = "✅" if actuals.outcome(*act) == actuals.outcome(bi, bj) else "❌"
                    real = f"<span class='real'>resultado real: {act[0]} × {act[1]} {ok}</span> · "
                tops = " · ".join(f"{i}-{j} ({p*100:.0f}%)" for i, j, p in pred["top_scores"])
                hl = actuals.outcome(*act) if act is not None else None
                bar = wdl_bar(pred["p_win"], pred["p_draw"], pred["p_loss"], hl)
                meta = (f"<div class='meta'>{real}prováveis: {tops} · gols esp. "
                        f"{pred['xg_a']:.1f}–{pred['xg_b']:.1f}</div>")
                st.markdown(f"<div class='match'>{teams}{pred_row}{meta}{bar}</div>",
                            unsafe_allow_html=True)

st.caption("Histórico: martj42/international_results · Resultados ao vivo: ESPN (sem chave) "
           "com fallback martj42 · Modelo Elo + Poisson, previsões out-of-sample.")
