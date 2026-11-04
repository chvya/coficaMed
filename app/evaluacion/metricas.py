"""Precisión, recall y F1 por caso, con coincidencia exacta o por categoría (3 primeros caracteres)."""
from __future__ import annotations


def _nivel(codigos: set[str], nivel: str) -> set[str]:
    codigos = {c.upper() for c in codigos}
    if nivel == "categoria":
        return {c.split(".")[0] for c in codigos}
    return codigos


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def evaluar(predichos: dict[str, set[str]], reales: dict[str, set[str]], nivel: str = "exacto") -> dict:
    """predichos y reales: id de caso -> conjunto de códigos."""
    tp = fp = fn = 0
    f1_casos = []
    for id_, real in reales.items():
        r = _nivel(real, nivel)
        p = _nivel(predichos.get(id_, set()), nivel)
        a, b, c = len(p & r), len(p - r), len(r - p)
        tp, fp, fn = tp + a, fp + b, fn + c
        f1_casos.append(prf(a, b, c)["f1"])
    micro = prf(tp, fp, fn)
    return {
        **micro,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1_macro": sum(f1_casos) / len(f1_casos) if f1_casos else 0.0,
        "casos": len(reales),
    }
