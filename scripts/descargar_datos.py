"""Descarga de Zenodo el corpus CodiEsp, la lista de códigos CIE-10 y NegEx-MES a data/raw/.

Uso: python scripts/descargar_datos.py"""
import sys
import zipfile
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"

RECURSOS = {
    "codiesp": "https://zenodo.org/api/records/3837305/files/codiesp.zip/content",
    "codigos": "https://zenodo.org/api/records/3706838/files/codiesp_codes.zip/content",
    "negex": "https://zenodo.org/api/records/2542567/files/PlanTL/NegEx-MES-v1.0.1.zip/content",
}


def descargar(nombre: str, url: str) -> None:
    destino = RAW / nombre
    if destino.exists() and any(destino.iterdir()):
        print(f"[ok] {nombre} ya existe en {destino}")
        return
    destino.mkdir(parents=True, exist_ok=True)
    zip_ruta = destino / f"{nombre}.zip"
    print(f"[..] descargando {nombre} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(zip_ruta, "wb") as f:
            for bloque in r.iter_bytes():
                f.write(bloque)
    with zipfile.ZipFile(zip_ruta) as z:
        z.extractall(destino)
    zip_ruta.unlink()
    print(f"[ok] {nombre} listo")


if __name__ == "__main__":
    for nombre, url in RECURSOS.items():
        try:
            descargar(nombre, url)
        except httpx.HTTPError as e:
            print(f"[error] no se pudo descargar {nombre}: {e}")
            sys.exit(1)
