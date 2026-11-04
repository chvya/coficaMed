"""Detección de negación e incertidumbre con NegEx y las frases de NegEx-MES."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import BASE_DIR
from .preprocesamiento import plegar

CARPETA = BASE_DIR / "data" / "negex"
VENTANA = 6  # máximo de palabras entre la frase de negación y la mención


@dataclass
class ResultadoNegacion:
    estado: str  # afirmado | negado | incierto
    disparador: str | None = None


def _leer(*nombres: str) -> list[str]:
    frases: list[str] = []
    for nombre in nombres:
        ruta = CARPETA / nombre
        if ruta.exists():
            for linea in ruta.read_text(encoding="utf8").splitlines():
                f = plegar(linea).strip()
                if f and f not in frases:
                    frases.append(f)
    return sorted(frases, key=len, reverse=True)


def _regex(frases: list[str]) -> re.Pattern | None:
    if not frases:
        return None
    alternativas = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in frases)
    return re.compile(rf"(?<![a-z0-9])(?:{alternativas})(?![a-z0-9])")


class DetectorNegacion:
    def __init__(self, ventana: int = VENTANA) -> None:
        self.ventana = ventana
        self.previa = _regex(_leer("negacion_previa.txt", "extra_negacion_previa.txt"))
        self.posterior = _regex(_leer("negacion_posterior.txt", "extra_negacion_posterior.txt"))
        self.pseudo = _regex(_leer("pseudonegacion.txt", "extra_pseudonegacion.txt"))
        self.conjuncion = _regex(_leer("conjunciones.txt", "extra_conjunciones.txt"))
        self.incertidumbre = _regex(_leer("extra_incertidumbre.txt"))

    @staticmethod
    def _spans(patron: re.Pattern | None, texto: str) -> list[tuple[int, int, str]]:
        if patron is None:
            return []
        return [(m.start(), m.end(), m.group()) for m in patron.finditer(texto)]

    @staticmethod
    def _solapa(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _palabras_entre(self, texto: str, a: int, b: int) -> int:
        return len(re.findall(r"[a-z0-9]+", texto[a:b]))

    def _corte_entre(self, texto: str, a: int, b: int, conjunciones: list[tuple[int, int, str]]) -> bool:
        if re.search(r"[:;.]|,\s*(?:y\s+)?con\b", texto[a:b]):  # ", con" abre una afirmación nueva
            return True
        return any(c[0] >= a and c[1] <= b for c in conjunciones)

    def evaluar(self, oracion: str, ini: int, fin: int) -> ResultadoNegacion:
        """Estado de la mención oracion[ini:fin]."""
        s = plegar(oracion)
        mencion = (ini, fin)
        pseudo = self._spans(self.pseudo, s)
        conj = self._spans(self.conjuncion, s)

        pseudo = pseudo + conj  # las conjunciones ("sin embargo") también anulan las frases que se solapan

        def libre(span: tuple[int, int, str]) -> bool:
            return not any(self._solapa((span[0], span[1]), (p[0], p[1])) for p in pseudo) and not self._solapa(
                (span[0], span[1]), mencion
            )

        previas = [t for t in self._spans(self.previa, s) if libre(t) and t[1] <= ini]
        posteriores = [t for t in self._spans(self.posterior, s) if libre(t) and t[0] >= fin]
        # la incertidumbre ('no se descarta') no se anula con las pseudo-negaciones
        inciertas = [
            t for t in self._spans(self.incertidumbre, s) if t[1] <= ini and not self._solapa((t[0], t[1]), mencion)
        ]

        for t in reversed(previas):  # la más cercana primero
            if (
                self._palabras_entre(s, t[1], ini) <= self.ventana
                and not self._corte_entre(s, t[1], ini, conj)
            ):
                return ResultadoNegacion("negado", t[2])
        for t in posteriores:
            if (
                self._palabras_entre(s, fin, t[0]) <= self.ventana
                and not self._corte_entre(s, fin, t[0], conj)
            ):
                return ResultadoNegacion("negado", t[2])
        for t in reversed(inciertas):
            if (
                self._palabras_entre(s, t[1], ini) <= self.ventana
                and not self._corte_entre(s, t[1], ini, conj)
            ):
                return ResultadoNegacion("incierto", t[2])
        return ResultadoNegacion("afirmado")
