"""Mide el recall@k de la recuperación con 300 anotaciones de dev.

Uso: python scripts/ajustar_recuperacion.py  (requiere Ollama con bge-m3 y los datos de descargar_datos.py)"""
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import app.pipeline.recuperacion as R  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.embeddings import vectorizar  # noqa: E402
from app.pipeline.preprocesamiento import expandir_abreviaturas, normalizar_frase  # noqa: E402

DEV_X = RAIZ / "data" / "raw" / "codiesp" / "final_dataset_v4_to_publish" / "dev" / "devX.tsv"
N, SEMILLA = 300, 1

# nombre, peso léxico, peso semántico, categorías agregadas, umbral léxico débil
CONFIGURACIONES = [
    ("Solo léxico (BM25)", 1.0, 0.0, 0, 0.0),
    ("Híbrido 0,7 léxico / 0,3 semántico", 0.7, 0.3, 3, 99),
    ("Híbrido 0,6 léxico / 0,4 semántico", 0.6, 0.4, 3, 99),
    ("Híbrido 0,5 léxico / 0,5 semántico", 0.5, 0.5, 3, 99),
    ("Híbrido 0,4 léxico / 0,6 semántico", 0.4, 0.6, 3, 99),
    ("Híbrido 0,3 léxico / 0,7 semántico", 0.3, 0.7, 3, 99),
    ("Léxico con refuerzo semántico (elegido)", 1.0, 0.1, 2, 0.3),
]


def main() -> None:
    r = R.Recuperador(True)
    if not r.semantica_disponible:
        sys.exit(f"Sin índice semántico: {r.motivo_sin_semantica}")
    items = []
    for linea in open(DEV_X, encoding="utf8"):
        p = linea.rstrip("\n").split("\t")
        if p[1] == "DIAGNOSTICO" and p[2].upper() in r.cat.descripcion:
            items.append((p[3], p[2].upper()))
    random.Random(SEMILLA).shuffle(items)
    items = items[:N]
    consultas = [expandir_abreviaturas(t) for t, _ in items]
    similitudes = vectorizar(consultas) @ r.categorias_matriz.T
    no_vista = [normalizar_frase(t) not in r.alias for t, _ in items]
    n_nuevas = sum(no_vista)

    resultados = []
    for nombre, pl, ps, ncat, umbral in CONFIGURACIONES:
        settings.peso_lexico, settings.peso_semantico = pl, ps
        R.CATEGORIAS_SEMANTICAS, R.UMBRAL_LEXICO_DEBIL = ncat, umbral
        usar = ps > 0
        total = {1: 0, 3: 0, 6: 0}
        nuevas = {1: 0, 3: 0, 6: 0}
        for i, (_, codigo) in enumerate(items):
            codigos = [c.codigo for c in r._una(consultas[i], similitudes[i] if usar else None, 6)]
            for k in total:
                if codigo in codigos[:k]:
                    total[k] += 1
                    nuevas[k] += no_vista[i]
        resultados.append(
            {
                "nombre": nombre,
                "todas": {k: round(v / len(items), 3) for k, v in total.items()},
                "no_vistas": {k: round(v / n_nuevas, 3) for k, v in nuevas.items()},
            }
        )
        print(nombre, resultados[-1]["todas"], resultados[-1]["no_vistas"])
    salida = settings.resultados_dir / "recuperacion_dev.json"
    salida.write_text(
        json.dumps({"anotaciones": len(items), "no_vistas": n_nuevas, "configuraciones": resultados}, ensure_ascii=False, indent=1),
        encoding="utf8",
    )
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
