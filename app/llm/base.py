"""Interfaz común de los proveedores de LLM."""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod


class ErrorLLM(Exception):
    """Error al comunicarse con el modelo o al interpretar su respuesta."""


class ErrorConexionLLM(ErrorLLM):
    """El proveedor no está disponible (apagado, sin clave o modelo inexistente)."""


class Metricas:
    """Acumula el uso de un proveedor: llamadas, tiempo y tokens."""

    def __init__(self) -> None:
        self.reiniciar()

    def reiniciar(self) -> None:
        self.llamadas = 0
        self.segundos = 0.0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    def registrar(self, segundos: float, entrada: int = 0, salida: int = 0) -> None:
        self.llamadas += 1
        self.segundos += segundos
        self.tokens_entrada += entrada or 0
        self.tokens_salida += salida or 0

    def como_dict(self) -> dict:
        return {
            "llamadas": self.llamadas,
            "segundos": round(self.segundos, 2),
            "tokens_entrada": self.tokens_entrada,
            "tokens_salida": self.tokens_salida,
        }


class ProveedorLLM(ABC):
    nombre: str = "base"
    modelo: str = ""

    def __init__(self) -> None:
        self.metricas = Metricas()

    @abstractmethod
    def generar_json(self, sistema: str, prompt: str, schema: dict, nombre: str) -> dict:
        """Devuelve un dict que respeta (en lo posible) el JSON Schema dado."""

    @abstractmethod
    def estado(self) -> dict:
        """Información para la interfaz: si está disponible y qué modelo usa."""

    def _medir(self, inicio: float, entrada: int = 0, salida: int = 0) -> None:
        self.metricas.registrar(time.perf_counter() - inicio, entrada, salida)


def extraer_json(texto: str) -> dict:
    """Interpreta JSON aunque el modelo lo envuelva en ```json ... ``` o agregue texto."""
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fin = texto.find("{"), texto.rfind("}")
        if inicio != -1 and fin > inicio:
            try:
                return json.loads(texto[inicio : fin + 1])
            except json.JSONDecodeError:
                pass
    reparado = _reparar_truncado(texto)
    if reparado is not None:
        return reparado
    raise ErrorLLM("El modelo no devolvió un JSON válido.")


def _reparar_truncado(texto: str) -> dict | None:
    """Recupera los elementos completos de un JSON cortado por el límite de tokens."""
    for pos in [m.start() for m in re.finditer(r"\}", texto)][::-1][:60]:
        try:
            return json.loads(texto[: pos + 1] + "]}")
        except json.JSONDecodeError:
            continue
    return None
