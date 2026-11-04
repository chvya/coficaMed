"""Proveedor para APIs con formato OpenAI (OpenAI, Gemini, Groq...). Usa JSON Schema si el servicio lo admite."""
from __future__ import annotations

import json
import time

import httpx

from ..config import settings
from .base import ErrorConexionLLM, ErrorLLM, ProveedorLLM, extraer_json


class ProveedorOpenAI(ProveedorLLM):
    nombre = "openai"

    def __init__(self) -> None:
        super().__init__()
        self.modelo = settings.openai_model
        self._formato_soportado = ["json_schema", "json_object", None]

    def _llamar(self, cuerpo: dict) -> httpx.Response:
        if not settings.openai_api_key:
            raise ErrorConexionLLM("Falta OPENAI_API_KEY en el archivo .env")
        try:
            return httpx.post(
                f"{settings.openai_base_url}/chat/completions",
                json=cuerpo,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=300,
            )
        except httpx.ConnectError as e:
            raise ErrorConexionLLM("Sin conexión con la API configurada en OPENAI_BASE_URL.") from e
        except httpx.TimeoutException as e:
            raise ErrorLLM("La API tardó demasiado en responder.") from e

    def generar_json(self, sistema: str, prompt: str, schema: dict, nombre: str) -> dict:
        sistema_json = (
            f"{sistema}\n\nResponde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        inicio = time.perf_counter()
        r = None
        for formato in list(self._formato_soportado):
            cuerpo: dict = {
                "model": self.modelo,
                "messages": [
                    {"role": "system", "content": sistema_json},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
            if formato == "json_schema":
                cuerpo["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": nombre, "schema": schema},
                }
            elif formato == "json_object":
                cuerpo["response_format"] = {"type": "json_object"}
            r = self._llamar(cuerpo)
            for intento in range(3):  # límite de solicitudes por minuto (frecuente en capas gratuitas)
                if r.status_code not in (429, 500, 502, 503):
                    break
                espera = float(r.headers.get("retry-after", 0) or 0) or 5 * (intento + 1)
                time.sleep(min(espera, 20))
                r = self._llamar(cuerpo)
            if r.status_code == 400 and formato is not None:
                # el servicio no admite este formato: se prueba el siguiente y se recuerda
                self._formato_soportado.remove(formato)
                continue
            break
        if r.status_code in (401, 403):
            raise ErrorConexionLLM("La API key no es válida o no tiene permiso.")
        if r.status_code == 404:
            raise ErrorConexionLLM(f"El modelo '{self.modelo}' no existe en este servicio. Revisa OPENAI_MODEL.")
        if r.status_code >= 400:
            raise ErrorLLM(f"La API respondió {r.status_code}: {r.text[:300]}")
        datos = r.json()
        uso = datos.get("usage", {})
        self._medir(inicio, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))
        try:
            contenido = datos["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ErrorLLM("Respuesta inesperada de la API.") from e
        return extraer_json(contenido or "")

    def estado(self) -> dict:
        configurado = bool(settings.openai_api_key)
        return {
            "disponible": configurado,
            "modelo": self.modelo,
            "mensaje": "API key configurada" if configurado else "Falta OPENAI_API_KEY en .env",
        }
