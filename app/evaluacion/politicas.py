"""Recalcula los códigos de un resultado guardado con otra política de negación o umbral de confianza."""
from __future__ import annotations

from ..pipeline.codificador import POLITICA_DEFECTO, motivo_exclusion

VARIANTES = {
    "sin_negacion": {"usar_negex": False, "usar_estado_llm": False, "excluir_inciertos": False},
    "solo_reglas": {"usar_negex": True, "usar_estado_llm": False, "excluir_inciertos": True},
    "solo_llm": {"usar_negex": False, "usar_estado_llm": True, "excluir_inciertos": True},
    "reglas_y_llm": dict(POLITICA_DEFECTO),
}


def puntaje_mencion(m: dict) -> float:
    """Puntaje de recuperación del código elegido (1,0 si viene del diccionario)."""
    if m.get("seleccion") == "diccionario":
        return 1.0
    return next((c["puntaje"] for c in m.get("candidatos", []) if c["codigo"] == m["codigo"]), 1.0)


def codigos_con_politica(resultado: dict, politica: dict, umbral: float = 0.0) -> set[str]:
    codigos = set()
    for m in resultado["menciones"]:
        if not m.get("codigo") or puntaje_mencion(m) < umbral:
            continue
        if not motivo_exclusion(m["estado_regla"], m.get("estado_llm", ""), politica):
            codigos.add(m["codigo"].upper())
    return codigos
