"""Catálogo CIE-10, índice BM25 y diccionario de expresiones del entrenamiento."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import snowballstemmer

from ..config import BASE_DIR
from .preprocesamiento import normalizar_frase, plegar, tokens_busqueda

CARPETA = BASE_DIR / "data" / "catalogo"
_stemmer = snowballstemmer.stemmer("spanish")


@dataclass
class Entrada:
    codigo: str
    texto: str
    fuente: str  # "catalogo" o "alias"
    n: int = 0  # frecuencia del alias en el entrenamiento


class Catalogo:
    def __init__(self) -> None:
        self.descripcion: dict[str, str] = {}
        for linea in open(CARPETA / "catalogo_cie10.tsv", encoding="utf8"):
            codigo, desc = linea.rstrip("\n").split("\t")
            self.descripcion[codigo] = desc
        self.categoria = lambda codigo: codigo.split(".")[0]
        self.categorias: list[str] = sorted(c for c in self.descripcion if "." not in c)
        self.por_categoria: dict[str, list[str]] = defaultdict(list)
        for c in self.descripcion:
            self.por_categoria[self.categoria(c)].append(c)
        sin_punto = sorted(c.replace(".", "") for c in self.descripcion)
        self.es_hoja = {c: not self._tiene_hijos(c.replace(".", ""), sin_punto) for c in self.descripcion}

        self.alias: list[Entrada] = []
        ruta_alias = CARPETA / "alias_entrenamiento.tsv"
        if ruta_alias.exists():
            for linea in open(ruta_alias, encoding="utf8"):
                alias, codigo, n = linea.rstrip("\n").split("\t")
                self.alias.append(Entrada(codigo, alias, "alias", int(n)))

    @staticmethod
    def _tiene_hijos(c: str, ordenados: list[str]) -> bool:
        import bisect

        i = bisect.bisect_right(ordenados, c)
        return i < len(ordenados) and ordenados[i].startswith(c)


class IndiceBM25:
    """Índice invertido BM25 sobre las entradas (descripciones y alias)."""

    K1, B = 1.2, 0.75

    def __init__(self, entradas: list[Entrada]) -> None:
        self.entradas = entradas
        self.invertido: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        provisional: dict[str, list[tuple[int, int]]] = defaultdict(list)
        longitudes = np.zeros(len(entradas), dtype=np.float32)
        for i, e in enumerate(entradas):
            toks = tokens_busqueda(e.texto)
            longitudes[i] = max(len(toks), 1)
            conteo: dict[str, int] = defaultdict(int)
            for t in toks:
                conteo[t] += 1
            for t, tf in conteo.items():
                provisional[t].append((i, tf))
        self.longitudes = longitudes
        self.promedio = float(longitudes.mean())
        n = len(entradas)
        self.idf: dict[str, float] = {}
        for t, lista in provisional.items():
            df = len(lista)
            self.idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
            docs = np.fromiter((d for d, _ in lista), dtype=np.int32, count=df)
            tfs = np.fromiter((f for _, f in lista), dtype=np.float32, count=df)
            self.invertido[t] = (docs, tfs)

    def puntuar(self, tokens: list[str]) -> tuple[np.ndarray, float]:
        """Puntaje BM25 de todas las entradas y la cota máxima teórica para normalizar a [0, 1]."""
        puntajes = np.zeros(len(self.entradas), dtype=np.float32)
        cota = 0.0
        for t in set(tokens):
            if t not in self.invertido:
                cota += 1.0  # término desconocido: cuenta como no cubierto
                continue
            idf = self.idf[t]
            cota += idf * (self.K1 + 1)
            docs, tfs = self.invertido[t]
            norma = self.K1 * (1 - self.B + self.B * self.longitudes[docs] / self.promedio)
            puntajes[docs] += idf * tfs * (self.K1 + 1) / (tfs + norma)
        return puntajes, max(cota, 1e-6)


@dataclass
class Coincidencia:
    ini: int
    fin: int
    texto: str
    codigo: str
    n: int


class Diccionario:
    """Busca en un texto las expresiones aprendidas del entrenamiento (coincidencia más larga)."""

    MAX_PALABRAS = 8

    def __init__(self, alias: list[Entrada], catalogo: Catalogo) -> None:
        mejores: dict[tuple, Entrada] = {}
        for e in alias:
            palabras = re.findall(r"[a-z0-9]+", plegar(e.texto))
            if not palabras or len(palabras) > self.MAX_PALABRAS:
                continue
            clave = tuple(_stemmer.stemWord(p) for p in palabras)
            if len(clave) == 1 and (len(clave[0]) < 3 or clave[0].isdigit()):
                continue
            actual = mejores.get(clave)
            if actual is None or e.n > actual.n:
                mejores[clave] = e
        self.tabla = {k: v for k, v in mejores.items()}
        self.catalogo = catalogo

    def buscar(self, texto: str) -> list[Coincidencia]:
        plano = plegar(texto)
        palabras = [(m.start(), m.end(), _stemmer.stemWord(m.group())) for m in re.finditer(r"[a-z0-9]+", plano)]
        coincidencias: list[Coincidencia] = []
        i = 0
        while i < len(palabras):
            hallada = None
            for largo in range(min(self.MAX_PALABRAS, len(palabras) - i), 0, -1):
                clave = tuple(p[2] for p in palabras[i : i + largo])
                if clave in self.tabla:
                    hallada = (largo, self.tabla[clave])
                    break
            if hallada:
                largo, entrada = hallada
                ini, fin = palabras[i][0], palabras[i + largo - 1][1]
                coincidencias.append(Coincidencia(ini, fin, texto[ini:fin], entrada.codigo, entrada.n))
                i += largo
            else:
                i += 1
        return coincidencias


@lru_cache(maxsize=1)
def cargar_catalogo() -> Catalogo:
    return Catalogo()


@lru_cache(maxsize=1)
def cargar_indice() -> IndiceBM25:
    cat = cargar_catalogo()
    entradas = [Entrada(c, d, "catalogo") for c, d in cat.descripcion.items()] + cat.alias
    return IndiceBM25(entradas)


@lru_cache(maxsize=1)
def cargar_diccionario() -> Diccionario:
    cat = cargar_catalogo()
    return Diccionario(cat.alias, cat)


def alias_exactos() -> dict[str, tuple[str, int]]:
    """Frase normalizada -> (código más frecuente, frecuencia)."""
    tabla: dict[str, tuple[str, int]] = {}
    for e in cargar_catalogo().alias:
        clave = normalizar_frase(e.texto)
        if clave not in tabla or e.n > tabla[clave][1]:
            tabla[clave] = (e.codigo, e.n)
    return tabla
