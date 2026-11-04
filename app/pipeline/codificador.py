"""Pipeline de codificación. Modos: reglas (diccionario + NegEx), llm e hibrido (ambos)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from ..llm import obtener_proveedor
from ..llm.base import ErrorLLM
from . import prompts
from .catalogo import cargar_catalogo, cargar_diccionario
from .negacion import DetectorNegacion
from .preprocesamiento import expandir_abreviaturas, normalizar_frase, oraciones, plegar
from .recuperacion import Candidato, Recuperador

Progreso = Callable[[str, str, str], None]

PASOS = [
    ("preprocesamiento", "Preprocesamiento", 2),
    ("extraccion", "Extracción de menciones", 3),
    ("negacion", "Detección de negación", 3),
    ("recuperacion", "Recuperación de candidatos CIE-10", 3),
    ("seleccion", "Selección del código", 3),
    ("consolidacion", "Consolidación con evidencia", 3),
]

POLITICA_DEFECTO = {"usar_negex": True, "usar_estado_llm": True, "excluir_inciertos": True, "puntaje_minimo": 0.0}
LOTE_SELECCION = 10
UMBRAL_RECHAZO = 0.6  # un "ninguno" del modelo solo se acepta si el mejor candidato tiene menos puntaje
CARACTERES_FRAGMENTO = 700  # tamaño aproximado de cada fragmento enviado al LLM


def motivo_exclusion(estado_regla: str, estado_llm: str, politica: dict) -> str | None:
    """Motivo por el que una mención no se codifica, o None si se codifica."""
    if politica["usar_negex"]:
        if estado_regla == "negado":
            return "negado"
        if estado_regla == "incierto" and politica["excluir_inciertos"]:
            return "incierto"
    if politica["usar_estado_llm"] and estado_llm:
        if estado_llm in ("negado", "descartado"):
            return estado_llm
        if estado_llm == "incierto" and politica["excluir_inciertos"]:
            return "incierto"
        if estado_llm == "familiar":
            return "familiar"
    return None


@dataclass
class Mencion:
    id: int
    frase: str
    ini: int
    fin: int
    oracion: str
    oracion_ini: int
    estado_llm: str = ""
    estado_regla: str = "afirmado"
    disparador: str | None = None
    candidatos: list[Candidato] = field(default_factory=list)
    codigo: str | None = None
    seleccion: str = ""  # "llm", "puntaje", "diccionario"

    def excluida(self, politica: dict) -> str | None:
        return motivo_exclusion(self.estado_regla, self.estado_llm, politica)

    def como_dict(self) -> dict:
        return {
            "id": self.id,
            "frase": self.frase,
            "ini": self.ini,
            "fin": self.fin,
            "oracion": self.oracion,
            "oracion_ini": self.oracion_ini,
            "estado_llm": self.estado_llm,
            "estado_regla": self.estado_regla,
            "disparador": self.disparador,
            "codigo": self.codigo,
            "seleccion": self.seleccion,
            "candidatos": [c.como_dict() for c in self.candidatos],
        }


def ubicar_frase(texto: str, frase: str, usadas: set[tuple[int, int]]) -> tuple[int, int]:
    """Posición de la frase en el texto (sin distinguir mayúsculas ni tildes); (-1, -1) si no aparece."""
    plano, buscada = plegar(texto), plegar(frase.strip())
    if not buscada:
        return -1, -1
    inicio = 0
    primera = None
    while True:
        i = plano.find(buscada, inicio)
        if i == -1:
            break
        span = (i, i + len(buscada))
        primera = primera or span
        if span not in usadas:
            return span
        inicio = i + 1
    return primera or (-1, -1)


def _oracion_de(spans: list[tuple[int, int]], posicion: int) -> tuple[int, int]:
    for ini, fin in spans:
        if ini <= posicion < fin:
            return ini, fin
    return spans[-1] if spans else (0, 0)


class Codificador:
    def __init__(self, usar_semantica: bool = True) -> None:
        self.cat = cargar_catalogo()
        self.negacion = DetectorNegacion()
        self._recuperador: Recuperador | None = None
        self._usar_semantica = usar_semantica

    @property
    def recuperador(self) -> Recuperador:
        if self._recuperador is None:
            self._recuperador = Recuperador(self._usar_semantica)
        return self._recuperador

    def codificar(
        self,
        nota: str,
        modo: str = "llm",
        proveedor: str | None = None,
        politica: dict | None = None,
        progreso: Progreso | None = None,
    ) -> dict:
        politica = {**POLITICA_DEFECTO, **(politica or {})}
        avisar = progreso or (lambda *_: None)
        pasos: list[dict] = []
        usa_llm = modo in ("llm", "hibrido")
        llm = obtener_proveedor(proveedor) if usa_llm else None
        if llm:
            llm.metricas.reiniciar()

        def paso(id_: str):
            return _Paso(id_, pasos, avisar)

        # 1. Preprocesamiento
        with paso("preprocesamiento") as p:
            spans = oraciones(nota)
            p.detalle = f"{len(spans)} oraciones"

        # 2. Extracción de menciones
        with paso("extraccion") as p:
            menciones = []
            if usa_llm:
                menciones = self._extraer_llm(nota, spans, llm)
                p.detalle = f"{len(menciones)} menciones extraídas por {llm.modelo}"
            if modo in ("reglas", "hibrido"):
                del_diccionario = self._extraer_diccionario(nota, spans, primer_id=len(menciones) + 1)
                menciones += del_diccionario
                p.detalle = (p.detalle + "; " if p.detalle else "") + f"{len(del_diccionario)} expresiones del diccionario"

        # 3. Negación
        with paso("negacion") as p:
            for m in menciones:
                res = self.negacion.evaluar(m.oracion, m.ini - m.oracion_ini, m.fin - m.oracion_ini)
                m.estado_regla, m.disparador = res.estado, res.disparador
            nega = sum(1 for m in menciones if m.estado_regla == "negado")
            p.detalle = f"{nega} menciones negadas por reglas"

        # 4-5. Recuperación y selección
        if usa_llm:
            with paso("recuperacion") as p:
                self._recuperar(menciones)
                sem = "con embeddings" if self.recuperador.semantica_disponible else "solo léxica"
                p.detalle = f"candidatos para {len(menciones)} menciones ({sem})"
            with paso("seleccion") as p:
                n_llm = self._seleccionar(menciones, llm, politica)
                p.detalle = f"{n_llm} menciones decididas por el LLM"
        else:
            with paso("recuperacion") as p:
                p.detalle = "no aplica: el diccionario ya entrega el código"
            with paso("seleccion") as p:
                p.detalle = "no aplica en la línea base"

        # 6. Consolidación
        with paso("consolidacion") as p:
            codigos, descartados = self.consolidar(menciones, politica)
            p.detalle = f"{len(codigos)} códigos sugeridos, {len(descartados)} menciones no codificadas"

        return {
            "modo": modo,
            "proveedor": llm.nombre if llm else "reglas",
            "modelo": llm.modelo if llm else "diccionario + NegEx",
            "politica": politica,
            "menciones": [m.como_dict() for m in menciones],
            "codigos": codigos,
            "descartados": descartados,
            "pasos": pasos,
            "metricas_llm": llm.metricas.como_dict() if llm else None,
            "semantica": bool(usa_llm and self.recuperador.semantica_disponible),
        }

    def _crear_mencion(self, id_: int, nota: str, spans, ini: int, fin: int, frase: str) -> Mencion:
        o_ini, o_fin = _oracion_de(spans, ini if ini >= 0 else 0)
        return Mencion(id_, frase, ini, fin, nota[o_ini:o_fin], o_ini)

    @staticmethod
    def _fragmentos(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Agrupa oraciones consecutivas en fragmentos de unos 700 caracteres."""
        fragmentos: list[tuple[int, int]] = []
        for ini, fin in spans:
            if fragmentos and fin - fragmentos[-1][0] <= CARACTERES_FRAGMENTO:
                fragmentos[-1] = (fragmentos[-1][0], fin)
            else:
                fragmentos.append((ini, fin))
        return fragmentos

    def _extraer_llm(self, nota: str, spans, llm) -> list[Mencion]:
        menciones: list[Mencion] = []
        usadas: set[tuple[int, int]] = set()
        vistas: set[tuple[str, str]] = set()
        for f_ini, f_fin in self._fragmentos(spans):
            datos = llm.generar_json(
                prompts.SISTEMA_EXTRACCION,
                prompts.prompt_extraccion(nota[f_ini:f_fin]),
                prompts.ESQUEMA_EXTRACCION,
                "registrar_menciones",
            )
            for item in datos.get("menciones", []):
                frase = str(item.get("frase", "")).strip()
                estado = str(item.get("estado", "afirmado")).strip().lower()
                if not frase or len(frase) > 120:
                    continue
                clave = (normalizar_frase(frase), estado)
                if clave in vistas:
                    continue
                vistas.add(clave)
                locales = {(a - f_ini, b - f_ini) for a, b in usadas if f_ini <= a < f_fin}
                ini, fin = ubicar_frase(nota[f_ini:f_fin], frase, locales)
                if ini == -1:
                    continue  # el modelo inventó o alteró la frase: no tiene evidencia en la nota
                ini, fin = ini + f_ini, fin + f_ini
                usadas.add((ini, fin))
                m = self._crear_mencion(len(menciones) + 1, nota, spans, ini, fin, nota[ini:fin])
                m.estado_llm = estado if estado in prompts.ESTADOS else "afirmado"
                menciones.append(m)
        return menciones

    def _extraer_diccionario(self, nota: str, spans, primer_id: int = 1) -> list[Mencion]:
        menciones = []
        for c in cargar_diccionario().buscar(nota):
            m = self._crear_mencion(primer_id + len(menciones), nota, spans, c.ini, c.fin, c.texto)
            m.codigo, m.seleccion = c.codigo, "diccionario"
            menciones.append(m)
        return menciones

    def _recuperar(self, menciones: list[Mencion]) -> None:
        menciones = [m for m in menciones if m.seleccion != "diccionario"]
        if not menciones:
            return
        consultas = [expandir_abreviaturas(m.frase) for m in menciones]
        for m, cands in zip(menciones, self.recuperador.candidatos(consultas)):
            m.candidatos = cands
            if cands:
                m.codigo, m.seleccion = cands[0].codigo, "puntaje"

    def _seleccionar(self, menciones: list[Mencion], llm, politica: dict) -> int:
        """El LLM elige el código de las menciones por codificar. Si la frase coincide con un alias, se omite la consulta."""
        pendientes = [
            m
            for m in menciones
            if m.candidatos and m.candidatos[0].fuente != "alias" and not m.excluida(politica)
        ]
        total = 0
        for i in range(0, len(pendientes), LOTE_SELECCION):
            lote = pendientes[i : i + LOTE_SELECCION]
            bloques = [
                {
                    "id": m.id,
                    "frase": expandir_abreviaturas(m.frase),
                    "oracion": re.sub(r"\s+", " ", m.oracion)[:160],
                    "candidatos": [(c.codigo, c.descripcion) for c in m.candidatos],
                }
                for m in lote
            ]
            try:
                datos = llm.generar_json(
                    prompts.SISTEMA_SELECCION,
                    prompts.prompt_seleccion(bloques),
                    prompts.ESQUEMA_SELECCION,
                    "registrar_codigos",
                )
            except ErrorLLM:
                continue  # se conserva el candidato de mayor puntaje
            decisiones = {}
            for d in datos.get("decisiones", []):
                try:
                    decisiones[int(d.get("id"))] = str(d.get("codigo", "")).strip().upper()
                except (TypeError, ValueError):
                    continue
            for m in lote:
                elegido = decisiones.get(m.id)
                if elegido == "NINGUNO":
                    # los modelos pequeños rechazan con facilidad: se conserva el candidato claro
                    if m.candidatos[0].puntaje < UMBRAL_RECHAZO:
                        m.codigo, m.seleccion = None, "llm"
                    total += 1
                elif elegido and any(c.codigo == elegido for c in m.candidatos):
                    m.codigo, m.seleccion = elegido, "llm"
                    total += 1
        return total

    def consolidar(self, menciones: list[Mencion], politica: dict) -> tuple[list[dict], list[dict]]:
        """Une las menciones en códigos únicos con su evidencia; separa las no codificadas."""
        por_codigo: dict[str, dict] = {}
        descartados: list[dict] = []
        for m in menciones:
            if not m.codigo:
                continue
            motivo = m.excluida(politica)
            evidencia = {"mencion": m.id, "frase": m.frase, "ini": m.ini, "fin": m.fin, "oracion": m.oracion}
            if motivo:
                descartados.append(
                    {
                        "codigo": m.codigo,
                        "descripcion": self.cat.descripcion.get(m.codigo, ""),
                        "motivo": motivo,
                        "disparador": m.disparador,
                        **evidencia,
                    }
                )
                continue
            puntaje = next((c.puntaje for c in m.candidatos if c.codigo == m.codigo), 1.0)
            if puntaje < politica.get("puntaje_minimo", 0.0):
                motivo = "confianza baja"
                descartados.append(
                    {
                        "codigo": m.codigo,
                        "descripcion": self.cat.descripcion.get(m.codigo, ""),
                        "motivo": motivo,
                        "disparador": None,
                        **evidencia,
                    }
                )
                continue
            fuente = next((c.fuente for c in m.candidatos if c.codigo == m.codigo), "diccionario")
            entrada = por_codigo.setdefault(
                m.codigo,
                {
                    "codigo": m.codigo,
                    "descripcion": self.cat.descripcion.get(m.codigo, ""),
                    "puntaje": puntaje,
                    "fuente": fuente,
                    "seleccion": m.seleccion,
                    "evidencias": [],
                },
            )
            entrada["puntaje"] = max(entrada["puntaje"], puntaje)
            entrada["evidencias"].append(evidencia)
        codigos = sorted(por_codigo.values(), key=lambda c: c["evidencias"][0]["ini"])
        for c in codigos:
            c["puntaje"] = round(min(c["puntaje"], 1.0), 3)
            c["confianza"] = "alta" if c["puntaje"] >= 0.7 else "media" if c["puntaje"] >= 0.45 else "baja"
        return codigos, descartados


class _Paso:
    """Contexto que mide la duración de un paso y avisa a la interfaz."""

    def __init__(self, id_: str, pasos: list[dict], avisar: Progreso) -> None:
        self.id = id_
        self.pasos = pasos
        self.avisar = avisar
        self.detalle = ""
        self.nombre, self.fase = next((n, f) for i, n, f in PASOS if i == id_)

    def __enter__(self):
        self.inicio = time.perf_counter()
        self.avisar(self.id, "iniciado", "")
        return self

    def __exit__(self, tipo, valor, tb):
        segundos = round(time.perf_counter() - self.inicio, 2)
        if tipo is None:
            self.pasos.append(
                {"id": self.id, "nombre": self.nombre, "fase": self.fase, "segundos": segundos, "detalle": self.detalle}
            )
            self.avisar(self.id, "listo", self.detalle)
        return False
