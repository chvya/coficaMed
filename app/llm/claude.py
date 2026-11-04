"""Proveedor Claude (API de Anthropic). La salida JSON se obtiene forzando una herramienta con el esquema."""
from __future__ import annotations

import time

from ..config import settings
from .base import ErrorConexionLLM, ErrorLLM, ProveedorLLM, extraer_json


class ProveedorClaude(ProveedorLLM):
    nombre = "claude"

    def __init__(self) -> None:
        super().__init__()
        self.modelo = settings.claude_model
        self._cliente = None

    def _cliente_api(self):
        if not settings.anthropic_api_key:
            raise ErrorConexionLLM("Falta ANTHROPIC_API_KEY en el archivo .env")
        if self._cliente is None:
            import anthropic

            self._cliente = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=3)
        return self._cliente

    def generar_json(self, sistema: str, prompt: str, schema: dict, nombre: str) -> dict:
        import anthropic

        inicio = time.perf_counter()
        try:
            respuesta = self._cliente_api().messages.create(
                model=self.modelo,
                max_tokens=4000,
                system=sistema,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": nombre,
                        "description": "Registra el resultado con la estructura indicada.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": nombre},
            )
        except anthropic.AuthenticationError as e:
            raise ErrorConexionLLM("La API key de Claude no es válida.") from e
        except anthropic.NotFoundError as e:
            raise ErrorConexionLLM(f"El modelo '{self.modelo}' no existe. Revisa CLAUDE_MODEL en .env") from e
        except anthropic.APIConnectionError as e:
            raise ErrorConexionLLM("Sin conexión con la API de Claude.") from e
        except anthropic.APIError as e:
            raise ErrorLLM(f"Error de la API de Claude: {e}") from e
        self._medir(inicio, respuesta.usage.input_tokens, respuesta.usage.output_tokens)
        for bloque in respuesta.content:
            if bloque.type == "tool_use":
                return dict(bloque.input)
        texto = "".join(b.text for b in respuesta.content if b.type == "text")
        return extraer_json(texto)

    def estado(self) -> dict:
        configurado = bool(settings.anthropic_api_key)
        return {
            "disponible": configurado,
            "modelo": self.modelo,
            "mensaje": "API key configurada" if configurado else "Falta ANTHROPIC_API_KEY en .env",
        }
