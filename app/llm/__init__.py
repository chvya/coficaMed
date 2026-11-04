"""Fábrica de proveedores de LLM: el "switch" local <-> API externa."""
from ..config import settings
from .base import ErrorConexionLLM, ErrorLLM, ProveedorLLM
from .claude import ProveedorClaude
from .ollama import ProveedorOllama
from .openai_compat import ProveedorOpenAI

PROVEEDORES = {
    "ollama": ProveedorOllama,
    "claude": ProveedorClaude,
    "openai": ProveedorOpenAI,
}

_instancias: dict[str, ProveedorLLM] = {}


def obtener_proveedor(nombre: str | None = None) -> ProveedorLLM:
    """Devuelve el proveedor pedido o, si no se indica, el de LLM_PROVIDER."""
    nombre = (nombre or settings.llm_provider).strip().lower()
    if nombre not in PROVEEDORES:
        raise ErrorLLM(f"Proveedor desconocido: '{nombre}'. Opciones: {', '.join(PROVEEDORES)}")
    if nombre not in _instancias:
        _instancias[nombre] = PROVEEDORES[nombre]()
    return _instancias[nombre]


__all__ = ["obtener_proveedor", "PROVEEDORES", "ProveedorLLM", "ErrorLLM", "ErrorConexionLLM"]
