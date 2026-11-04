"""Proveedor local con Ollama: chat con salida JSON restringida por el esquema."""
from __future__ import annotations

import json
import time

import httpx

from ..config import settings
from .base import ErrorConexionLLM, ErrorLLM, ProveedorLLM, extraer_json


class ProveedorOllama(ProveedorLLM):
    nombre = "ollama"

    def __init__(self) -> None:
        super().__init__()
        self.host = settings.ollama_host
        self.modelo = settings.ollama_model

    def _chat(self, sistema: str, prompt: str, formato: dict | None) -> str:
        cuerpo: dict = {
            "model": self.modelo,
            "messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0,  # extracción fiel y reproducible
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": 900,  # evita generaciones sin fin en CPU
            },
            "keep_alive": "30m",
        }
        if formato is not None:
            cuerpo["format"] = formato
        inicio = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", json=cuerpo, timeout=settings.ollama_timeout)
        except httpx.ConnectError as e:
            raise ErrorConexionLLM("No se pudo conectar con Ollama. ¿Está abierto? (ollama serve)") from e
        except httpx.TimeoutException as e:
            raise ErrorLLM("Ollama tardó demasiado en responder.") from e
        if r.status_code == 404:
            raise ErrorConexionLLM(f"El modelo '{self.modelo}' no está descargado. Ejecuta: ollama pull {self.modelo}")
        if r.status_code >= 400:
            raise ErrorLLM(f"Ollama respondió {r.status_code}: {r.text[:300]}")
        datos = r.json()
        self._medir(inicio, datos.get("prompt_eval_count", 0), datos.get("eval_count", 0))
        return datos.get("message", {}).get("content", "")

    def generar_json(self, sistema: str, prompt: str, schema: dict, nombre: str) -> dict:
        sistema_json = (
            f"{sistema}\n\nResponde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        ultimo_error: Exception | None = None
        for _ in range(2):  # un reintento si el JSON llega mal formado
            try:
                return extraer_json(self._chat(sistema_json, prompt, schema))
            except ErrorConexionLLM:
                raise
            except ErrorLLM as e:
                ultimo_error = e
        raise ultimo_error or ErrorLLM("Error desconocido")

    def estado(self) -> dict:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3)
            instalados = [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError):
            return {"disponible": False, "modelo": self.modelo, "mensaje": "Ollama no está en ejecución"}

        def presente(nombre: str) -> bool:
            base = nombre if ":" in nombre else f"{nombre}:latest"
            return any(m in (nombre, base) for m in instalados)

        if not presente(self.modelo):
            return {"disponible": False, "modelo": self.modelo, "mensaje": f"Falta: ollama pull {self.modelo}"}
        mensaje = "Listo"
        if settings.embeddings_provider == "ollama" and not presente(settings.embeddings_model):
            mensaje = f"Listo (sin embeddings: ollama pull {settings.embeddings_model})"
        return {"disponible": True, "modelo": self.modelo, "mensaje": mensaje}
