"""Embeddings locales con Ollama (por defecto bge-m3), también cuando el LLM es una API."""
from __future__ import annotations

import httpx
import numpy as np

from ..config import settings


def embeddings_disponibles() -> bool:
    if settings.embeddings_provider != "ollama":
        return False
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3)
        nombres = [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return False
    modelo = settings.embeddings_model
    base = modelo if ":" in modelo else f"{modelo}:latest"
    return any(n in (modelo, base) for n in nombres)


def vectorizar(textos: list[str], lote: int = 16) -> np.ndarray | None:
    """Devuelve una matriz (n, d) normalizada, o None si Ollama no responde."""
    vectores: list[list[float]] = []
    try:
        for i in range(0, len(textos), lote):
            r = httpx.post(
                f"{settings.ollama_host}/api/embed",
                json={"model": settings.embeddings_model, "input": textos[i : i + lote], "keep_alive": "30m"},
                timeout=settings.ollama_timeout,
            )
            if r.status_code >= 400:
                return None
            vectores.extend(r.json().get("embeddings", []))
    except httpx.HTTPError:
        return None
    if len(vectores) != len(textos):
        return None
    m = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(normas, 1e-9, None)
