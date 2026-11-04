"""Evalúa un sistema (reglas, ollama, claude u openai) y guarda el resultado de cada caso.

Uso: python scripts/evaluar.py --sistema ollama --conjunto evaluacion --n 15
Si se interrumpe, al repetir el comando continúa donde quedó."""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.config import settings  # noqa: E402
from app.evaluacion.metricas import evaluar  # noqa: E402
from app.evaluacion.politicas import VARIANTES, codigos_con_politica  # noqa: E402
from app.pipeline.codificador import Codificador  # noqa: E402


def cargar_casos(conjunto: str, n: int | None) -> list[dict]:
    ruta = RAIZ / "data" / "casos" / f"{conjunto}.jsonl"
    casos = [json.loads(linea) for linea in open(ruta, encoding="utf8")]
    return casos[:n] if n else casos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sistema", required=True, choices=["reglas", "ollama", "claude", "openai"])
    ap.add_argument("--conjunto", default="evaluacion", choices=["evaluacion", "dev", "test", "train"])
    ap.add_argument("--n", type=int, default=None, help="usar solo los primeros n casos")
    ap.add_argument("--etiqueta", default=None, help="nombre del archivo de salida (por defecto, el sistema)")
    ap.add_argument("--sin-semantica", action="store_true", help="recuperación solo léxica")
    args = ap.parse_args()

    etiqueta = args.etiqueta or args.sistema
    salida = settings.resultados_dir / f"{etiqueta}_{args.conjunto}.jsonl"
    hechos = {}
    if salida.exists():
        for linea in open(salida, encoding="utf8"):
            r = json.loads(linea)
            hechos[r["id"]] = r

    casos = cargar_casos(args.conjunto, args.n)
    codificador = Codificador(usar_semantica=not args.sin_semantica)
    modo = "reglas" if args.sistema == "reglas" else "llm"
    proveedor = None if modo == "reglas" else args.sistema

    pendientes = [c for c in casos if c["id"] not in hechos]
    print(f"{len(casos)} casos, {len(hechos)} ya evaluados, {len(pendientes)} pendientes -> {salida.name}")
    with open(salida, "a", encoding="utf8", newline="\n") as f:
        for i, caso in enumerate(pendientes, 1):
            inicio = time.perf_counter()
            try:
                resultado = codificador.codificar(caso["texto"], modo=modo, proveedor=proveedor)
            except Exception as e:  # un caso fallido no detiene la evaluación
                print(f"  [{caso['id']}] error: {e}")
                if "conectar" in str(e).lower() or "api key" in str(e).lower():
                    sys.exit(1)
                continue
            segundos = round(time.perf_counter() - inicio, 1)
            registro = {"id": caso["id"], "gold": caso["codigos"], "segundos": segundos, "resultado": resultado}
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush()
            hechos[caso["id"]] = registro
            n_cod = len(resultado["codigos"])
            print(f"  [{i}/{len(pendientes)}] {caso['id']}: {n_cod} códigos en {segundos} s", flush=True)

    reales = {c["id"]: set(c["codigos"]) for c in casos if c["id"] in hechos}
    print("\nResultados (exacto / categoría):")
    for nombre, politica in VARIANTES.items():
        pred = {i: codigos_con_politica(hechos[i]["resultado"], politica) for i in reales}
        e, c = evaluar(pred, reales), evaluar(pred, reales, "categoria")
        print(
            f"  {nombre:14s} P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f}"
            f"   | categoría F1={c['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
