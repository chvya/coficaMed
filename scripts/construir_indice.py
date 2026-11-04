"""Calcula los embeddings de las categorías CIE-10 (una vez, unos 10 minutos sin GPU).

Uso: python scripts/construir_indice.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.recuperacion import construir_indice_categorias  # noqa: E402

if __name__ == "__main__":
    inicio = time.time()
    ruta = construir_indice_categorias(verbose=True)
    print(f"Índice guardado en {ruta} ({time.time() - inicio:.0f} s)")
