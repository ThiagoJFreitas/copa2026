# ⚽ Copa do Mundo 2026 — Previsão da Fase de Grupos

App em Streamlit que prevê os **72 jogos da fase de grupos** (sem mata-mata) da Copa de 2026,
combinando **Elo** + **Poisson** (ataque/defesa ajustados por adversário, nível de gols
calibrado), treinados no histórico real de jogos de seleções. As previsões são
**out-of-sample** (o modelo só vê jogos anteriores ao início da Copa), o que permite
**comparar previsão × resultado real** dos jogos que já aconteceram. Nomes em português e
bandeiras das seleções.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

O navegador abre em `http://localhost:8501`. Não precisa de nenhuma chave de API.

## Tipos de modelo (presets)

Na barra lateral, escolha um **Tipo de modelo** — cada um é uma combinação coerente de
parâmetros (peso Elo×Poisson, memória, ano de corte, nível de gols):

| Preset | Caráter |
|--------|---------|
| ⚖️ **Equilibrado** (padrão) | 50/50 Elo e Poisson, ~5 anos de memória. Bom padrão geral (~2,6 gols/jogo). |
| 🔥 **Forma recente** | Memória curta (2 anos), só jogos desde 2014: prioriza o momento. |
| 📜 **Histórico (Elo puro)** | Só Elo, memória longa: favorece potências tradicionais. |
| ⚽ **Ofensivo (mais gols)** | Mais Poisson e nível de gols alto: placares elásticos (~3,2 gols/jogo). |
| 🛡️ **Defensivo (poucos gols)** | Nível de gols baixo e mais Elo: jogos truncados (~2,0 gols/jogo). |
| 🛠️ **Personalizado** | Expõe todos os sliders (incl. **nível de gols**) — equivale ao ajuste manual. |

> **Nível de gols e placares:** o placar exibido é o resultado exato mais provável (modal),
> mas como o número de gols *exato* mais provável é sempre baixo, cada jogo também mostra os
> **gols esperados** (λ). O parâmetro *nível de gols* calibra o realismo — o padrão (~1,45)
> gera médias de Copa (~2,6 gols/jogo) e placares variados (3-0, 2-1) nos confrontos desiguais.

## Comparação previsão × realidade

A aba **📊 Previsão × Realidade** lista os jogos da fase de grupos já disputados, com placar
previsto vs real, acerto do resultado, placar exato, erro médio de gols e a probabilidade
média que o modelo deu ao resultado que de fato ocorreu. Inclui também um **gráfico de
calibração** (probabilidade prevista × frequência real).

### Fonte dos resultados (com prioridade)

1. **ESPN** — API pública de placar (`site.api.espn.com/.../soccer/fifa.world/scoreboard`),
   **sem chave**. É a fonte **ao vivo** dos resultados de 2026, uma requisição por data,
   cacheada até você clicar em **🔄**.
2. **martj42** — fallback (costuma ter alguns dias de atraso).

A ESPN tem precedência sobre o martj42, e os resultados ainda ausentes no martj42 são
**injetados no histórico de treino** — então o modo rolling e as projeções de classificação
passam a refletir os jogos mais recentes. A aba mostra de qual fonte vieram os resultados.

### Modo rolling

O toggle **Modo rolling** (barra lateral) muda como as previsões são feitas:

- **Desligado (padrão):** todas as previsões usam um modelo fixo treinado só com jogos
  anteriores ao início da Copa (out-of-sample puro).
- **Ligado:** cada jogo é previsto por um modelo treinado com tudo que aconteceu **até a
  véspera daquele jogo** — então a previsão de um jogo do round 2 já incorpora os resultados
  do round 1. As projeções de classificação passam a refletir os jogos já disputados.

## Fonte dos dados

[`martj42/international_results`](https://github.com/martj42/international_results) — CSV no
GitHub, sem chave de API, com todos os jogos internacionais de 1872 ao presente
(data, seleções, placar, torneio, cidade, e flag de jogo em campo neutro).
O CSV é baixado e fica em cache no diretório temporário do SO por 24h
(botão **🔄 Atualizar histórico** força novo download).

## Como o modelo funciona

| Componente | O que faz |
|------------|-----------|
| **Elo ponderado** (`model.compute_elo`) | Itera os jogos em ordem cronológica e ajusta o rating de cada seleção. O ajuste é maior em jogos importantes (Copa/eliminatórias > amistosos) e proporcional à diferença de gols. Dá força e probabilidade V-E-D. |
| **Poisson** (`model.compute_strengths`) | Estima força de **ataque** e **defesa** de cada seleção, **ajustadas pela força do adversário** (ponto-fixo iterativo estilo Dixon-Coles) e ponderadas por **recência** e **importância** do torneio. O ajuste por adversário evita inflar o ataque de times que goleiam adversários fracos (ex.: eliminatórias da CONCACAF). Gera a matriz de placares → placar mais provável e probabilidades. |
| **Nível de gols** (`goal_scale`) | Multiplica os gols esperados (λ) para calibrar o realismo dos placares. Cada preset usa um valor; o padrão (~1,45) dá médias de Copa (~2,6 gols/jogo). |
| **Mistura** (`predict_match`) | As probabilidades V-E-D finais são uma média ponderada dos dois modelos (*Peso Elo × Poisson*). O placar exibido é o pico da matriz de Poisson **dentro** do resultado previsto; os **gols esperados** (λ) também são mostrados. |
| **Simulação de grupo** (`simulate_group`) | Monte Carlo: sorteia placares de Poisson para os 6 jogos, N vezes, e conta com que frequência cada seleção termina no **top 2** (% de classificação), com desempate por pontos → saldo → gols pró. |

## Parâmetros (modo Personalizado)

- **Peso Elo × Poisson** — 0 = só placares (Poisson), 1 = só ratings (Elo).
- **Meia-vida da recência** — quanto menor, mais os jogos recentes dominam.
- **Considerar jogos a partir de** — ano de corte do histórico.
- **Nível de gols** — calibra quantos gols saem (placares mais ou menos elásticos).
- **Simulações por grupo** — mais simulações = % de classificação mais estável.

## Arquivos

```
data.py       download + cache do CSV (histórico/resultados)
teams.py      nomes canônicos (FIFA/API/PT) e normalização
display.py    nomes em português + bandeiras (imagens flagcdn)
fixtures.py   12 grupos e os 72 jogos da Copa 2026
espn.py       resultados ao vivo da Copa 2026 via API pública da ESPN (sem chave)
actuals.py    mescla resultados (ESPN/martj42) e casa com os fixtures (a "verdade")
model.py      Elo + Poisson (ajustado por adversário) + simulação de grupo (vetorizada)
app.py        interface Streamlit (presets, abas de grupo, Previsão × Realidade)
```

> `players_api.py` (cliente API-Football para notas de jogadores) foi descontinuado do app —
> as notas frequentemente faltavam para muitas seleções e não agregavam. O arquivo pode ser
> ignorado/removido.

> Os fixtures e grupos refletem o sorteio final (Washington DC, 05/12/2025), com os
> vencedores de repescagem já resolvidos.
