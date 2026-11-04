"""Configuración leída del archivo .env (ver .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(valor: str | None, defecto: bool = False) -> bool:
    if valor is None:
        return defecto
    return valor.strip().lower() in {"1", "true", "si", "sí", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        # --- Proveedor activo: "ollama", "claude" u "openai" ---
        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        # --- Ollama (modelo local) ---
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "900"))

        # --- Embeddings (recuperación semántica) ---
        # "ollama" usa un modelo de embeddings local; "lexico" usa solo BM25.
        self.embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "ollama").strip().lower()
        self.embeddings_model = os.getenv("EMBEDDINGS_MODEL", "bge-m3")

        # --- Claude (API de Anthropic) ---
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

        # --- API con formato OpenAI (OpenAI, Gemini, Groq, OpenRouter...) ---
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # --- Recuperación y decisión ---
        self.top_k_candidatos = int(os.getenv("TOP_K_CANDIDATOS", "6"))
        self.peso_semantico = float(os.getenv("PESO_SEMANTICO", "0.1"))
        self.peso_lexico = float(os.getenv("PESO_LEXICO", "1.0"))

        # --- Rutas ---
        self.data_dir = BASE_DIR / "data"
        self.indices_dir = self.data_dir / "indices"
        self.resultados_dir = self.data_dir / "resultados"
        for carpeta in (self.indices_dir, self.resultados_dir):
            carpeta.mkdir(parents=True, exist_ok=True)

        self.puerto = int(os.getenv("PUERTO", "8000"))


settings = Settings()
