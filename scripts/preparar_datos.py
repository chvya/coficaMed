"""Genera catálogo, diccionario, listas de negación y casos a partir de data/raw/.

Uso: python scripts/preparar_datos.py"""
import collections
import json
import random
import re
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
CORPUS = RAW / "codiesp" / "final_dataset_v4_to_publish"
CODIGOS = RAW / "codigos" / "codiesp_codes" / "codiesp-D_codes.tsv"
NEGEX = next((RAW / "negex").glob("PlanTL-NegEx-MES-*")) / "smn" / "config_files" / "spa" / "without_lemma"

SEMILLA = 42
TAMANO_MUESTRA = 40
MAX_CARACTERES_CODIGO = 6  # el 99,9 % de los códigos reales del corpus tiene 6 o menos

# Frases de NegEx-MES que son traducciones literales sin sentido clínico en español.
FRASES_DESCARTADAS = {
    "ro", "r / o", "gobierna al paciente", "expulsó al paciente por", "más bien que",
    "no poder", "no pueden ver", "hacer una prueba por", "revisado para", "evaluar por",
    "declina", "no ser", "no ver", "no sentir", "nada nuevo", "no apreciar", "no exhibir",
    "no demostrar", "no revelar", "no saber de", "no me quejo de", "paciente no era",
    "nunca tuve", "nunca desarrollado", "no se sabe que tiene", "resuelto",
}


def texto_plano(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def preparar_catalogo() -> set[str]:
    salida = RAIZ / "data" / "catalogo" / "catalogo_cie10.tsv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    codigos = set()
    with open(CODIGOS, encoding="utf8") as f, open(salida, "w", encoding="utf8", newline="\n") as g:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 2:
                continue
            codigo, descripcion = partes[0].strip().upper(), texto_plano(partes[1])
            if len(codigo.replace(".", "")) > MAX_CARACTERES_CODIGO:
                continue
            g.write(f"{codigo}\t{descripcion}\n")
            codigos.add(codigo)
    print(f"catálogo: {len(codigos)} códigos")
    return codigos


def leer_casos(conjunto: str) -> list[dict]:
    base = CORPUS / conjunto
    codigos = collections.defaultdict(set)
    with open(base / f"{conjunto}D.tsv", encoding="utf8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) >= 2:
                codigos[partes[0]].add(partes[1].strip().upper())
    casos = []
    for archivo in sorted((base / "text_files").glob("*.txt")):
        casos.append(
            {
                "id": archivo.stem,
                "texto": archivo.read_text(encoding="utf8").strip(),
                "codigos": sorted(codigos.get(archivo.stem, set())),
            }
        )
    return casos


def preparar_alias(catalogo: set[str]) -> None:
    """Expresiones diagnósticas del conjunto de entrenamiento (CodiEsp-X) con su código."""
    cuenta: collections.Counter = collections.Counter()
    with open(CORPUS / "train" / "trainX.tsv", encoding="utf8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 4 or partes[1] != "DIAGNOSTICO":
                continue
            codigo, alias = partes[2].strip().upper(), texto_plano(partes[3]).lower()
            if codigo in catalogo and len(alias) >= 3:
                cuenta[(alias, codigo)] += 1
    salida = RAIZ / "data" / "catalogo" / "alias_entrenamiento.tsv"
    with open(salida, "w", encoding="utf8", newline="\n") as g:
        for (alias, codigo), n in sorted(cuenta.items()):
            g.write(f"{alias}\t{codigo}\t{n}\n")
    print(f"alias de entrenamiento: {len(cuenta)}")


def preparar_negex() -> None:
    destino = RAIZ / "data" / "negex"
    destino.mkdir(parents=True, exist_ok=True)
    archivos = {
        "neg-phrases.txt": "negacion_previa.txt",
        "post-neg-phrases.txt": "negacion_posterior.txt",
        "pseudo-neg-phrases.txt": "pseudonegacion.txt",
        "conjunctions.txt": "conjunciones.txt",
    }
    for origen, nombre in archivos.items():
        frases = []
        for linea in open(NEGEX / origen, encoding="utf8"):
            frase = texto_plano(linea).lower()
            if not frase or frase.startswith(":") or frase in FRASES_DESCARTADAS or frase in frases:
                continue
            frases.append(frase)
        (destino / nombre).write_text("\n".join(frases) + "\n", encoding="utf8", newline="\n")
        print(f"negex {nombre}: {len(frases)} frases")


def preparar_casos() -> None:
    destino = RAIZ / "data" / "casos"
    destino.mkdir(parents=True, exist_ok=True)
    por_conjunto = {c: leer_casos(c) for c in ("train", "dev", "test")}
    for conjunto, casos in por_conjunto.items():
        with open(destino / f"{conjunto}.jsonl", "w", encoding="utf8", newline="\n") as f:
            for caso in casos:
                f.write(json.dumps(caso, ensure_ascii=False) + "\n")
        print(f"casos {conjunto}: {len(casos)}")
    test = list(por_conjunto["test"])
    random.Random(SEMILLA).shuffle(test)
    with open(destino / "evaluacion.jsonl", "w", encoding="utf8", newline="\n") as f:
        for caso in test[:TAMANO_MUESTRA]:
            f.write(json.dumps(caso, ensure_ascii=False) + "\n")
    print(f"muestra de evaluación: {TAMANO_MUESTRA} casos del conjunto test (semilla {SEMILLA})")


if __name__ == "__main__":
    catalogo = preparar_catalogo()
    preparar_alias(catalogo)
    preparar_negex()
    preparar_casos()
