"""Exibição: nomes das seleções em português e bandeiras (imagens).

Usamos imagens do flagcdn.com (não emoji), porque no Windows os emojis de
bandeira costumam não renderizar (aparecem como "BR" em vez de 🇧🇷).
"""

from __future__ import annotations

from teams import canonical

# nome canônico (inglês) -> nome em português
TEAM_PT = {
    "Mexico": "México", "South Africa": "África do Sul", "South Korea": "Coreia do Sul",
    "Czech Republic": "Tchéquia", "Canada": "Canadá",
    "Bosnia and Herzegovina": "Bósnia e Herzegovina", "Qatar": "Catar",
    "Switzerland": "Suíça", "Brazil": "Brasil", "Morocco": "Marrocos", "Haiti": "Haiti",
    "Scotland": "Escócia", "United States": "Estados Unidos", "Paraguay": "Paraguai",
    "Australia": "Austrália", "Turkey": "Turquia", "Germany": "Alemanha",
    "Ecuador": "Equador", "Ivory Coast": "Costa do Marfim", "Curaçao": "Curaçao",
    "Netherlands": "Países Baixos", "Japan": "Japão", "Sweden": "Suécia",
    "Tunisia": "Tunísia", "Belgium": "Bélgica", "Egypt": "Egito", "Iran": "Irã",
    "New Zealand": "Nova Zelândia", "Spain": "Espanha", "Uruguay": "Uruguai",
    "Saudi Arabia": "Arábia Saudita", "Cape Verde": "Cabo Verde", "France": "França",
    "Senegal": "Senegal", "Iraq": "Iraque", "Norway": "Noruega", "Argentina": "Argentina",
    "Algeria": "Argélia", "Austria": "Áustria", "Jordan": "Jordânia",
    "Portugal": "Portugal", "Colombia": "Colômbia", "DR Congo": "RD Congo",
    "Uzbekistan": "Uzbequistão", "England": "Inglaterra", "Croatia": "Croácia",
    "Ghana": "Gana", "Panama": "Panamá",
}

# nome canônico -> código do flagcdn (ISO 3166-1 alpha-2; subdivisões com prefixo gb-)
TEAM_CODE = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Ecuador": "ec", "Ivory Coast": "ci", "Curaçao": "cw",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz", "Spain": "es",
    "Uruguay": "uy", "Saudi Arabia": "sa", "Cape Verde": "cv", "France": "fr",
    "Senegal": "sn", "Iraq": "iq", "Norway": "no", "Argentina": "ar", "Algeria": "dz",
    "Austria": "at", "Jordan": "jo", "Portugal": "pt", "Colombia": "co", "DR Congo": "cd",
    "Uzbekistan": "uz", "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}


def pt(name: str) -> str:
    """Nome da seleção em português."""
    return TEAM_PT.get(canonical(name), name)


def flag_url(name: str, w: int = 48, h: int = 36) -> str:
    """URL da imagem da bandeira (flagcdn). Vazio se não houver código."""
    code = TEAM_CODE.get(canonical(name))
    return f"https://flagcdn.com/{w}x{h}/{code}.png" if code else ""


def img(name: str, height: int = 14) -> str:
    """Tag <img> inline da bandeira (para markdown com unsafe_allow_html)."""
    url = flag_url(name)
    if not url:
        return ""
    return (f"<img src='{url}' height='{height}' "
            f"style='vertical-align:middle;border-radius:2px;margin-right:4px'>")


def label_html(name: str, height: int = 14) -> str:
    """Bandeira + nome em português, para markdown."""
    return f"{img(name, height)}{pt(name)}"


# ---------------------------------------------------------------------------
# Layout responsivo (mobile-first): CSS + tabelas HTML + cartões de jogo
# ---------------------------------------------------------------------------
TABLE_CSS = """
<style>
.block-container{padding-top:2.4rem;}
@media (max-width:640px){.block-container{padding-left:.7rem;padding-right:.7rem;}}
/* tabelas com rolagem horizontal suave (não estouram o layout no celular) */
.cup-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid rgba(128,128,128,.25);
  border-radius:8px;margin:.2rem 0 .5rem;}
table.cup{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap;}
table.cup th,table.cup td{padding:6px 9px;border-bottom:1px solid rgba(128,128,128,.18);text-align:center;}
table.cup th{font-size:10px;font-weight:700;letter-spacing:.04em;color:#8a9099;
  text-transform:uppercase;background:rgba(128,128,128,.10);}
table.cup tr:last-child td{border-bottom:none;}
table.cup td.nm,table.cup th.nm{text-align:left;font-weight:600;}
table.cup td.pos{color:#8a9099;width:1.6em;}
table.cup img{height:14px;width:auto;border-radius:2px;vertical-align:middle;margin-right:6px;}
table.cup tr.q td{background:rgba(46,125,50,.14);}
.qbar{display:inline-block;width:54px;height:13px;background:rgba(128,128,128,.22);border-radius:4px;
  overflow:hidden;vertical-align:middle;margin-right:5px;}
.qbar>i{display:block;height:100%;background:#2e7d32;}
/* cartões de jogo */
.match{padding:11px 2px;border-bottom:1px solid rgba(128,128,128,.16);}
.match .teams{display:flex;align-items:center;flex-wrap:wrap;gap:5px;font-size:14px;line-height:1.5;}
.match .teams .vs{color:#8a9099;margin:0 2px;}
.match .pred{margin:7px 0 5px;display:flex;align-items:center;gap:9px;flex-wrap:wrap;}
.match .pred .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#8a9099;}
.match .pred .sc{font-weight:800;font-size:21px;letter-spacing:1.5px;color:#fff;
  background:rgba(46,125,50,.22);border:1px solid rgba(46,125,50,.55);
  padding:2px 13px;border-radius:9px;}
.match .pred .pwin{font-weight:700;color:#e6e8eb;font-size:15px;}
.match .meta{color:#8a9099;font-size:12px;margin-top:3px;}
.match .real{color:#e0b020;font-weight:600;}
.match img{height:16px;border-radius:2px;vertical-align:middle;}
</style>
"""


def html_table(headers, rows) -> str:
    """Tabela HTML responsiva. headers=[(label, css)], rows=[(cells, row_css)]
    onde cells=[(html, css)]."""
    thead = "".join(f"<th class='{c}'>{h}</th>" for h, c in headers)
    trs = ""
    for cells, rcls in rows:
        tds = "".join(f"<td class='{c}'>{v}</td>" for v, c in cells)
        trs += f"<tr class='{rcls}'>{tds}</tr>"
    return (f"<div class='cup-wrap'><table class='cup'><thead><tr>{thead}</tr>"
            f"</thead><tbody>{trs}</tbody></table></div>")


def qbar(pct: float) -> str:
    """Barra de progresso compacta + porcentagem (para % de classificação)."""
    w = max(0.0, min(100.0, pct))
    return f"<span class='qbar'><i style='width:{w:.0f}%'></i></span>{pct:.0f}%"
