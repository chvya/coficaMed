"""Recuperación de códigos candidatos: BM25 sobre el catálogo y embeddings de categorías."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import settings
from ..llm.embeddings import embeddings_disponibles, vectorizar
from .catalogo import alias_exactos, cargar_catalogo, cargar_indice
from .preprocesamiento import normalizar_frase, tokens_busqueda

SIM_MIN, SIM_RANGO = 0.40, 0.40  # similitud coseno que equivale a 0 y a 1
BONUS_ALIAS = 0.25
PENALIZACION_NO_HOJA = 0.08
CATEGORIAS_SEMANTICAS = 2  # categorías cuyos códigos se agregan como candidatos
UMBRAL_LEXICO_DEBIL = 0.3  # solo se agregan si la mejor coincidencia léxica es menor a este valor
CANDIDATOS_LEXICOS = 40


@dataclass
class Candidato:
    codigo: str
    descripcion: str
    puntaje: float
    lexico: float
    semantico: float
    fuente: str

    def como_dict(self) -> dict:
        return {
            "codigo": self.codigo,
            "descripcion": self.descripcion,
            "puntaje": round(self.puntaje, 3),
            "lexico": round(self.lexico, 3),
            "semantico": round(self.semantico, 3),
            "fuente": self.fuente,
        }


def ruta_indice() -> Path:
    modelo = re.sub(r"[^a-zA-Z0-9]+", "_", settings.embeddings_model)
    return settings.indices_dir / f"categorias_{modelo}.npz"


def construir_indice_categorias(verbose: bool = False) -> Path:
    """Calcula y guarda los embeddings de las categorías de 3 caracteres."""
    cat = cargar_catalogo()
    textos = [cat.descripcion[c] for c in cat.categorias]
    parcial = ruta_indice().with_suffix(".parcial.npy")
    bloques: list[np.ndarray] = []
    hecho = 0
    if parcial.exists():
        previo = np.load(parcial)
        bloques.append(previo)
        hecho = len(previo)
        if verbose:
            print(f"Reanudando desde {hecho}/{len(textos)}")
    paso = 64
    while hecho < len(textos):
        vec = vectorizar(textos[hecho : hecho + paso])
        if vec is None:
            raise RuntimeError("No se pudo calcular embeddings: ¿está abierto Ollama y descargado el modelo?")
        bloques.append(vec)
        hecho += len(vec)
        np.save(parcial, np.vstack(bloques))
        if verbose:
            print(f"  {hecho}/{len(textos)}", flush=True)
    matriz = np.vstack(bloques).astype(np.float16)
    np.savez_compressed(ruta_indice(), codigos=np.array(cat.categorias), matriz=matriz)
    parcial.unlink(missing_ok=True)
    return ruta_indice()


class Recuperador:
    def __init__(self, usar_semantica: bool = True) -> None:
        self.cat = cargar_catalogo()
        self.indice = cargar_indice()
        self.alias = alias_exactos()
        self.codigo_de_entrada = [e.codigo for e in self.indice.entradas]
        self.entradas_de_codigo: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(self.indice.entradas):
            self.entradas_de_codigo[e.codigo].append(i)
        self.categorias_matriz: np.ndarray | None = None
        self.categorias_codigos: list[str] = []
        self.motivo_sin_semantica = ""
        if usar_semantica:
            self._cargar_semantica()
        else:
            self.motivo_sin_semantica = "desactivada"

    def _cargar_semantica(self) -> None:
        ruta = ruta_indice()
        if not ruta.exists():
            self.motivo_sin_semantica = "índice no construido (python scripts/construir_indice.py)"
            return
        if not embeddings_disponibles():
            self.motivo_sin_semantica = "modelo de embeddings no disponible en Ollama"
            return
        datos = np.load(ruta)
        self.categorias_codigos = [str(c) for c in datos["codigos"]]
        self.categorias_matriz = datos["matriz"].astype(np.float32)

    @property
    def semantica_disponible(self) -> bool:
        return self.categorias_matriz is not None

    def candidatos(self, consultas: list[str], k: int | None = None) -> list[list[Candidato]]:
        """Candidatos para cada consulta (frase ya con abreviaturas desarrolladas)."""
        k = k or settings.top_k_candidatos
        similitudes = None
        if self.semantica_disponible and consultas:
            vectores = vectorizar(consultas)
            if vectores is not None:
                similitudes = vectores @ self.categorias_matriz.T
        return [
            self._una(consulta, None if similitudes is None else similitudes[i], k)
            for i, consulta in enumerate(consultas)
        ]

    def _una(self, consulta: str, sim: np.ndarray | None, k: int) -> list[Candidato]:
        puntajes, cota = self.indice.puntuar(tokens_busqueda(consulta))
        cobertura = np.clip(puntajes / cota, 0, 1)

        # léxico: mejores entradas -> mejor puntaje por código
        mejores = np.argpartition(-cobertura, min(CANDIDATOS_LEXICOS, len(cobertura) - 1))[:CANDIDATOS_LEXICOS]
        lex: dict[str, float] = {}
        for i in mejores:
            if cobertura[i] > 0:
                c = self.codigo_de_entrada[i]
                lex[c] = max(lex.get(c, 0.0), float(cobertura[i]))

        # semántico: valor por categoría y miembros de las mejores categorías
        sem_cat: dict[str, float] = {}
        if sim is not None:
            valores = np.clip((sim - SIM_MIN) / SIM_RANGO, 0, 1)
            sem_cat = {self.categorias_codigos[j]: float(valores[j]) for j in np.argsort(-sim)[:10]}
            if max(lex.values(), default=0.0) < UMBRAL_LEXICO_DEBIL:
                for j in np.argsort(-sim)[:CATEGORIAS_SEMANTICAS]:
                    for codigo in self.cat.por_categoria[self.categorias_codigos[j]]:
                        if codigo not in lex:
                            lex[codigo] = max(float(cobertura[i]) for i in self.entradas_de_codigo[codigo])

        alias = self.alias.get(normalizar_frase(consulta))
        if alias and alias[0] in self.cat.descripcion and alias[0] not in lex:
            lex[alias[0]] = 1.0

        pl, ps = (settings.peso_lexico, settings.peso_semantico) if sim is not None else (1.0, 0.0)
        resultado: list[Candidato] = []
        for codigo, valor_lex in lex.items():
            valor_sem = sem_cat.get(self.cat.categoria(codigo), 0.0)
            puntaje = pl * valor_lex + ps * valor_sem
            es_alias = bool(alias and alias[0] == codigo)
            if es_alias:
                puntaje += BONUS_ALIAS
            if not self.cat.es_hoja.get(codigo, True):
                puntaje -= PENALIZACION_NO_HOJA
            if es_alias:
                fuente = "alias"
            elif valor_lex > 0 and valor_sem >= 0.3:
                fuente = "híbrido"
            elif valor_sem > valor_lex:
                fuente = "semántico"
            else:
                fuente = "léxico"
            resultado.append(
                Candidato(codigo, self.cat.descripcion[codigo], puntaje, valor_lex, valor_sem, fuente)
            )
        resultado.sort(key=lambda c: c.puntaje, reverse=True)
        return resultado[:k]
