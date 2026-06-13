"""Fase de grupos da Copa do Mundo de 2026 — grupos e os 72 jogos.

Fonte: sorteio final (Washington DC, 05/12/2025) e páginas por grupo da
Wikipédia, com os vencedores de repescagem já resolvidos (o torneio começou
em 11/06/2026). Os nomes seguem a nomenclatura do dataset martj42, exceto
onde traduzidos por teams.py.
"""

from __future__ import annotations

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Cada jogo: (grupo, mandante, visitante, data ISO)
MATCHES = [
    # Grupo A
    ("A", "Mexico", "South Africa", "2026-06-11"),
    ("A", "South Korea", "Czech Republic", "2026-06-11"),
    ("A", "Czech Republic", "South Africa", "2026-06-18"),
    ("A", "Mexico", "South Korea", "2026-06-18"),
    ("A", "Czech Republic", "Mexico", "2026-06-24"),
    ("A", "South Africa", "South Korea", "2026-06-24"),
    # Grupo B
    ("B", "Canada", "Bosnia and Herzegovina", "2026-06-12"),
    ("B", "Qatar", "Switzerland", "2026-06-13"),
    ("B", "Switzerland", "Bosnia and Herzegovina", "2026-06-18"),
    ("B", "Canada", "Qatar", "2026-06-18"),
    ("B", "Switzerland", "Canada", "2026-06-24"),
    ("B", "Bosnia and Herzegovina", "Qatar", "2026-06-24"),
    # Grupo C
    ("C", "Brazil", "Morocco", "2026-06-13"),
    ("C", "Haiti", "Scotland", "2026-06-13"),
    ("C", "Scotland", "Morocco", "2026-06-19"),
    ("C", "Brazil", "Haiti", "2026-06-19"),
    ("C", "Scotland", "Brazil", "2026-06-24"),
    ("C", "Morocco", "Haiti", "2026-06-24"),
    # Grupo D
    ("D", "United States", "Paraguay", "2026-06-12"),
    ("D", "Australia", "Turkey", "2026-06-13"),
    ("D", "United States", "Australia", "2026-06-19"),
    ("D", "Turkey", "Paraguay", "2026-06-19"),
    ("D", "Turkey", "United States", "2026-06-25"),
    ("D", "Paraguay", "Australia", "2026-06-25"),
    # Grupo E
    ("E", "Germany", "Curaçao", "2026-06-14"),
    ("E", "Ivory Coast", "Ecuador", "2026-06-14"),
    ("E", "Germany", "Ivory Coast", "2026-06-20"),
    ("E", "Ecuador", "Curaçao", "2026-06-20"),
    ("E", "Curaçao", "Ivory Coast", "2026-06-25"),
    ("E", "Ecuador", "Germany", "2026-06-25"),
    # Grupo F
    ("F", "Netherlands", "Japan", "2026-06-14"),
    ("F", "Sweden", "Tunisia", "2026-06-14"),
    ("F", "Netherlands", "Sweden", "2026-06-20"),
    ("F", "Tunisia", "Japan", "2026-06-20"),
    ("F", "Japan", "Sweden", "2026-06-25"),
    ("F", "Tunisia", "Netherlands", "2026-06-25"),
    # Grupo G
    ("G", "Belgium", "Egypt", "2026-06-15"),
    ("G", "Iran", "New Zealand", "2026-06-15"),
    ("G", "Belgium", "Iran", "2026-06-21"),
    ("G", "New Zealand", "Egypt", "2026-06-21"),
    ("G", "Egypt", "Iran", "2026-06-26"),
    ("G", "New Zealand", "Belgium", "2026-06-26"),
    # Grupo H
    ("H", "Spain", "Cape Verde", "2026-06-15"),
    ("H", "Saudi Arabia", "Uruguay", "2026-06-15"),
    ("H", "Spain", "Saudi Arabia", "2026-06-21"),
    ("H", "Uruguay", "Cape Verde", "2026-06-21"),
    ("H", "Cape Verde", "Saudi Arabia", "2026-06-26"),
    ("H", "Uruguay", "Spain", "2026-06-26"),
    # Grupo I
    ("I", "France", "Senegal", "2026-06-16"),
    ("I", "Iraq", "Norway", "2026-06-16"),
    ("I", "France", "Iraq", "2026-06-22"),
    ("I", "Norway", "Senegal", "2026-06-22"),
    ("I", "Norway", "France", "2026-06-26"),
    ("I", "Senegal", "Iraq", "2026-06-26"),
    # Grupo J
    ("J", "Argentina", "Algeria", "2026-06-16"),
    ("J", "Austria", "Jordan", "2026-06-16"),
    ("J", "Argentina", "Austria", "2026-06-22"),
    ("J", "Jordan", "Algeria", "2026-06-22"),
    ("J", "Algeria", "Austria", "2026-06-27"),
    ("J", "Jordan", "Argentina", "2026-06-27"),
    # Grupo K
    ("K", "Portugal", "DR Congo", "2026-06-17"),
    ("K", "Uzbekistan", "Colombia", "2026-06-17"),
    ("K", "Portugal", "Uzbekistan", "2026-06-23"),
    ("K", "Colombia", "DR Congo", "2026-06-23"),
    ("K", "Colombia", "Portugal", "2026-06-27"),
    ("K", "DR Congo", "Uzbekistan", "2026-06-27"),
    # Grupo L
    ("L", "England", "Croatia", "2026-06-17"),
    ("L", "Ghana", "Panama", "2026-06-17"),
    ("L", "England", "Ghana", "2026-06-23"),
    ("L", "Panama", "Croatia", "2026-06-23"),
    ("L", "Panama", "England", "2026-06-27"),
    ("L", "Croatia", "Ghana", "2026-06-27"),
]


def matches_for_group(group: str):
    """Retorna os jogos de um grupo específico."""
    return [m for m in MATCHES if m[0] == group]
