"""Normalização de nomes de seleções.

Os nomes usados nos fixtures da FIFA/sorteio nem sempre batem com os nomes
do dataset martj42. Este módulo traduz de um para o outro.

A maioria dos nomes é idêntica; só listamos aqui os que diferem. A função
`to_dataset_name` faz fallback para o próprio nome quando não há mapeamento.
"""

from __future__ import annotations

import unicodedata

# Aliases (FIFA / API-Football / variações) -> nomenclatura usada nos fixtures
# (que coincide com o dataset martj42). Chaves comparadas após remover acentos.
FIFA_TO_DATASET = {
    "USMNT": "United States",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "South-Korea": "South Korea",
    "Turkiye": "Turkey",          # "Türkiye" cai aqui após remover acento
    "Cote d'Ivoire": "Ivory Coast",  # idem "Côte d'Ivoire"
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "Congo-DR": "DR Congo",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",  # grafia da ESPN
    "Korea DPR": "North Korea",
    "China PR": "China",
}


def _fold(s: str) -> str:
    """Remove acentos para comparação robusta de nomes."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def to_dataset_name(name: str) -> str:
    """Converte um nome de fixture para o nome equivalente no dataset."""
    return FIFA_TO_DATASET.get(name, name)


def canonical(name: str) -> str:
    """Nome canônico (nomenclatura dos fixtures), tolerante a acentos/variações.

    Usado para casar resultados vindos do API-Football com os fixtures.
    """
    n = name.strip()
    if n in FIFA_TO_DATASET:
        return FIFA_TO_DATASET[n]
    folded = _fold(n)
    if folded in FIFA_TO_DATASET:
        return FIFA_TO_DATASET[folded]
    return n
