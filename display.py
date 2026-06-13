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
