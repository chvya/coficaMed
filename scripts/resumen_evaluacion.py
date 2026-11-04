"""Reúne los resultados de evaluar.py en data/resultados/resumen.json (lo lee la pestaña Evaluación).

Uso: python scripts/resumen_evaluacion.py"""
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.config import settings  # noqa: E402
from app.pipeline.negacion import DetectorNegacion  # noqa: E402
from app.pipeline.preprocesamiento import oraciones  # noqa: E402
from app.evaluacion.metricas import evaluar, prf  # noqa: E402
from app.evaluacion.politicas import VARIANTES, codigos_con_politica  # noqa: E402

NOMBRES = {
    "reglas": "Diccionario + NegEx (sin modelo de lenguaje)",
    "ollama": "Modelo local",
    "claude": "API externa Claude",
    "openai": "API externa (Gemini)",
}
PREFIJO_HIBRIDO = "hibrido-"
UMBRALES = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]
TITULOS = {
    "evaluacion": "Muestra de evaluación del conjunto test",
    "test": "Conjunto test completo",
    "dev": "Conjunto dev completo",
}
ORDEN = ["reglas", "ollama", "claude", "openai"]


def leer(ruta: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(l) for l in open(ruta, encoding="utf8"))}


def cargar_revision() -> list[dict]:
    ruta = settings.data_dir / "casos" / "revision_negacion.json"
    return json.loads(ruta.read_text(encoding="utf8"))["anotaciones"] if ruta.exists() else []


def codigos_ajustados(casos: dict[str, set[str]], revision: list[dict]) -> dict[str, set[str]]:
    """Códigos reales sin los que solo aparecen en menciones negadas o inciertas."""
    afirmados: dict[str, set[str]] = defaultdict(set)
    for a in revision:
        if a["estado"] == "afirmado":
            afirmados[a["caso"]].add(a["codigo"])
    anotados = {(a["caso"], a["codigo"]) for a in revision}
    return {
        i: {c for c in codigos if c in afirmados[i] or (i, c) not in anotados}
        for i, codigos in casos.items()
    }


def exactitud_negacion(revision: list[dict]) -> dict:
    """Compara NegEx (aplicado a la frase de cada anotación) con la revisión manual."""
    textos = {json.loads(l)["id"]: json.loads(l)["texto"] for l in open(settings.data_dir / "casos" / "evaluacion.jsonl", encoding="utf8")}
    detector = DetectorNegacion()
    tp = fp = fn = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in revision:
        texto = textos[a["caso"]]
        o_ini, o_fin = next((i, f) for i, f in oraciones(texto) if i <= a["ini"] < f)
        res = detector.evaluar(texto[o_ini:o_fin], a["ini"] - o_ini, min(a["fin"], o_fin) - o_ini)
        confusion[a["estado"]][res.estado] += 1
        manual, regla = a["estado"] != "afirmado", res.estado != "afirmado"
        tp, fp, fn = tp + (manual and regla), fp + (regla and not manual), fn + (manual and not regla)
    return {
        "anotaciones": len(revision),
        "no_afirmadas_revision": sum(1 for a in revision if a["estado"] != "afirmado"),
        **{k: v for k, v in prf(tp, fp, fn).items()},
        "tp": tp, "fp": fp, "fn": fn,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def _orden(etiqueta: str) -> tuple:
    base = etiqueta.removeprefix(PREFIJO_HIBRIDO)
    return (ORDEN.index(base) if base in ORDEN else 99, etiqueta.startswith(PREFIJO_HIBRIDO))


def _combinar(reglas: dict, llm: dict) -> dict:
    """Resultado del modo híbrido: menciones del diccionario y del modelo juntas."""
    menciones = llm["resultado"]["menciones"] + reglas["resultado"]["menciones"]
    return {
        **llm,
        "segundos": round(llm["segundos"] + reglas["segundos"], 1),
        "resultado": {**llm["resultado"], "menciones": menciones, "modelo": "diccionario + " + llm["resultado"]["modelo"]},
    }


def main() -> None:
    grupos: dict[str, dict[str, dict]] = defaultdict(dict)
    for ruta in sorted(settings.resultados_dir.glob("*_*.jsonl")):
        etiqueta, conjunto = ruta.stem.rsplit("_", 1)
        grupos[conjunto][etiqueta] = leer(ruta)

    revision = cargar_revision()
    conjuntos = []
    for conjunto, sistemas in grupos.items():
        ids = set.intersection(*(set(r) for r in sistemas.values()))
        if not ids:
            continue
        reales = {i: {c.upper() for c in next(iter(sistemas.values()))[i]["gold"]} for i in ids}
        sistemas = dict(sistemas)
        if "reglas" in sistemas:  # modo híbrido = menciones del diccionario + menciones del modelo
            for etiqueta in [e for e in sistemas if e != "reglas"]:
                sistemas[PREFIJO_HIBRIDO + etiqueta] = {i: _combinar(sistemas["reglas"][i], sistemas[etiqueta][i]) for i in ids}
        reales_ajustados = codigos_ajustados(reales, revision) if conjunto == "evaluacion" and revision else None
        lista = []
        for etiqueta in sorted(sistemas, key=_orden):
            filas = [sistemas[etiqueta][i] for i in ids]
            sensibilidad = []
            for umbral in UMBRALES:
                pred = {r["id"]: codigos_con_politica(r["resultado"], VARIANTES["sin_negacion"], umbral) for r in filas}
                e = evaluar(pred, reales)
                sensibilidad.append({"umbral": umbral, **{k: e[k] for k in ("precision", "recall", "f1")}})
            variantes, ajustado = {}, {}
            for nombre, politica in VARIANTES.items():
                pred = {r["id"]: codigos_con_politica(r["resultado"], politica) for r in filas}
                variantes[nombre] = {"exacto": evaluar(pred, reales), "categoria": evaluar(pred, reales, "categoria")}
                if reales_ajustados:
                    ajustado[nombre] = {
                        "exacto": evaluar(pred, reales_ajustados),
                        "categoria": evaluar(pred, reales_ajustados, "categoria"),
                    }
            uso = [r["resultado"].get("metricas_llm") or {} for r in filas]
            lista.append(
                {
                    "id": etiqueta,
                    "nombre": NOMBRES.get(etiqueta) or "Híbrido: diccionario + " + NOMBRES.get(etiqueta[len(PREFIJO_HIBRIDO):], "").replace("Modelo local", "modelo local"),
                    "modelo": filas[0]["resultado"]["modelo"],
                    "seg_caso": round(sum(r["segundos"] for r in filas) / len(filas), 1),
                    "tokens_entrada": sum(u.get("tokens_entrada", 0) for u in uso) / len(filas),
                    "tokens_salida": sum(u.get("tokens_salida", 0) for u in uso) / len(filas),
                    "variantes": variantes,
                    "variantes_ajustado": ajustado or None,
                    "sensibilidad_umbral": sensibilidad,
                }
            )
        conjuntos.append(
            {
                "id": conjunto,
                "titulo": TITULOS.get(conjunto, conjunto),
                "casos": len(ids),
                "nota": "Todos los sistemas se comparan sobre los mismos casos. Los códigos reales de CodiEsp "
                "incluyen menciones negadas, por eso el filtro de negación reduce el recall frente a ese criterio.",
                "sistemas": lista,
            }
        )
    conjuntos.sort(key=lambda c: ["evaluacion", "test", "dev"].index(c["id"]) if c["id"] in TITULOS else 9)
    salida = settings.resultados_dir / "resumen.json"
    salida.write_text(
        json.dumps(
            {
                "generado": date.today().isoformat(),
                "conjuntos": conjuntos,
                "negacion": exactitud_negacion(revision) if revision else None,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf8",
    )
    print(f"Resumen guardado en {salida}")
    for c in conjuntos:
        print(f"\n{c['titulo']} ({c['casos']} casos)")
        for s in c["sistemas"]:
            e = s["variantes"]["sin_negacion"]["exacto"]
            n = s["variantes"]["reglas_y_llm"]["exacto"]
            print(f"  {s['id']:8s} sin filtro P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f} | con filtro F1={n['f1']:.3f}")


if __name__ == "__main__":
    main()
