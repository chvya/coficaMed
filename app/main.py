"""API REST y servidor de la interfaz."""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import BASE_DIR, settings
from .llm import PROVEEDORES, obtener_proveedor
from .llm.base import ErrorConexionLLM, ErrorLLM
from .pipeline.catalogo import cargar_diccionario, cargar_indice
from .pipeline.codificador import PASOS, Codificador

STATIC = BASE_DIR / "app" / "static"

app = FastAPI(title="CodificaMed", description="Codificación automática CIE-10 de notas clínicas con PLN")
codificador = Codificador()
cola = ThreadPoolExecutor(max_workers=1)  # un análisis a la vez: el modelo local usa toda la CPU
trabajos: dict[str, dict] = {}


class Peticion(BaseModel):
    nota: str = Field(min_length=10, max_length=8000)
    modo: Literal["llm", "hibrido", "reglas"] = "llm"
    proveedor: Literal["ollama", "claude", "openai"] | None = None
    politica: dict | None = None


def _calentar() -> None:
    """Carga los índices al arrancar para que el primer análisis no espere."""
    cargar_indice()
    cargar_diccionario()
    codificador.recuperador  # noqa: B018


@app.on_event("startup")
def al_iniciar() -> None:
    threading.Thread(target=_calentar, daemon=True).start()


@app.get("/api/estado")
def estado() -> dict:
    proveedores = {}
    for nombre in PROVEEDORES:
        try:
            proveedores[nombre] = obtener_proveedor(nombre).estado()
        except ErrorLLM as e:
            proveedores[nombre] = {"disponible": False, "modelo": "", "mensaje": str(e)}
    r = codificador._recuperador
    return {
        "proveedores": proveedores,
        "predeterminado": settings.llm_provider,
        "indices_listos": r is not None,
        "semantica": bool(r and r.semantica_disponible),
        "motivo_sin_semantica": r.motivo_sin_semantica if r else "",
        "embeddings_modelo": settings.embeddings_model,
    }


@app.get("/api/ejemplos")
def ejemplos() -> list[dict]:
    lista = []
    ruta = BASE_DIR / "data" / "ejemplos" / "notas_ficticias.json"
    if ruta.exists():
        for e in json.loads(ruta.read_text(encoding="utf8")):
            lista.append({**e, "origen": "ficticio"})
    ruta = BASE_DIR / "data" / "casos" / "evaluacion.jsonl"
    if ruta.exists():
        for i, linea in enumerate(open(ruta, encoding="utf8")):
            if i >= 5:
                break
            c = json.loads(linea)
            lista.append(
                {
                    "id": c["id"],
                    "titulo": f"Caso CodiEsp {c['id']}",
                    "texto": c["texto"],
                    "esperados": c["codigos"],
                    "origen": "codiesp",
                }
            )
    return lista


def _ejecutar(id_trabajo: str, p: Peticion) -> None:
    trabajo = trabajos[id_trabajo]
    trabajo["estado"] = "procesando"
    trabajo["inicio"] = time.time()

    def progreso(paso: str, estado_paso: str, detalle: str) -> None:
        trabajo["pasos"][paso] = {"estado": estado_paso, "detalle": detalle, "t": round(time.time() - trabajo["inicio"], 1)}

    try:
        trabajo["resultado"] = codificador.codificar(
            p.nota, modo=p.modo, proveedor=p.proveedor, politica=p.politica, progreso=progreso
        )
        trabajo["estado"] = "listo"
    except ErrorConexionLLM as e:
        trabajo["estado"], trabajo["error"] = "error", str(e)
    except Exception as e:  # noqa: BLE001 - se informa al usuario sin tumbar el servidor
        trabajo["estado"], trabajo["error"] = "error", f"Error inesperado: {e}"
    trabajo["segundos"] = round(time.time() - trabajo["inicio"], 1)


@app.post("/api/codificar")
def codificar(p: Peticion) -> dict:
    id_trabajo = uuid.uuid4().hex[:10]
    trabajos[id_trabajo] = {
        "estado": "en_cola",
        "pasos": {},
        "pasos_definidos": [{"id": i, "nombre": n, "fase": f} for i, n, f in PASOS],
        "modo": p.modo,
        "proveedor": p.proveedor or settings.llm_provider,
    }
    cola.submit(_ejecutar, id_trabajo, p)
    return {"id": id_trabajo}


@app.get("/api/trabajos/{id_trabajo}")
def ver_trabajo(id_trabajo: str) -> dict:
    if id_trabajo not in trabajos:
        raise HTTPException(404, "Trabajo no encontrado")
    t = trabajos[id_trabajo]
    if t["estado"] == "procesando":
        t = {**t, "transcurrido": round(time.time() - t["inicio"], 1)}
    return t


@app.get("/api/evaluacion")
def evaluacion() -> dict:
    ruta = settings.resultados_dir / "resumen.json"
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf8"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def inicio() -> FileResponse:
    return FileResponse(STATIC / "index.html")
