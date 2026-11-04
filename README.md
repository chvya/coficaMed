# CodificaMed

Prototipo que recibe una nota clínica en texto libre y sugiere códigos CIE-10, mostrando la frase de la nota que justifica cada código. Distingue los diagnósticos afirmados de los negados, descartados, inciertos y familiares.

Tarea de la materia Base de Conocimiento (Procesamiento de Lenguaje Natural), ESPOCH - FIE - EIS. Autor: Angel D. Gadvay.

## Entregables

| Entregable | Ubicación |
|---|---|
| Código fuente | Este repositorio: carpetas `app/`, `scripts/`, `tests/` y `data/` |
| Informe ejecutivo | [`informe/Informe_CodificaMed.docx`](informe/Informe_CodificaMed.docx) |
| Video de demostración | [Ver en Google Drive](https://drive.google.com/file/d/1RKDOEHkMCXERuPl9nBm0Fm6QFg_mUV_1/view?usp=sharing) |

## Qué hace

1. **Preprocesamiento:** segmenta la nota en oraciones y desarrolla abreviaturas clínicas (HTA, DM2, EPOC...) para buscar en el catálogo.
2. **Extracción de menciones:** un modelo de lenguaje (local o API) identifica enfermedades, síntomas y hallazgos, y el diccionario aprendido del entrenamiento aporta las expresiones conocidas.
3. **Negación:** el algoritmo NegEx con las frases de NegEx-MES detecta negación e incertidumbre; el modelo también indica el estado de cada mención.
4. **Recuperación (RAG):** busca candidatos en el catálogo CIE-10 con BM25 y embeddings (bge-m3).
5. **Selección:** el modelo elige el código entre los candidatos.
6. **Consolidación:** une las menciones en códigos únicos con su evidencia.

El modelo se cambia por configuración entre **Ollama (local)**, **Claude** y cualquier **API con formato OpenAI** (OpenAI, Gemini, Groq, OpenRouter).

## Requisitos

- Python 3.11 o superior.
- [Ollama](https://ollama.com/download) para el modelo local y los embeddings.
- Unos 6 GB libres en disco. Con 8 GB de RAM, cierre otras aplicaciones antes de analizar con el modelo local.

## Instalación

**Windows:** ejecute `instalar.bat`. **Linux / Mac:** `chmod +x instalar.sh iniciar.sh && ./instalar.sh`.

El script crea el entorno virtual `.venv`, instala las librerías y copia `.env.example` como `.env`. Después:

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

El índice semántico del catálogo ya viene en `data/indices/`. Para regenerarlo (unos 10 minutos sin GPU): `python scripts/construir_indice.py`.

Si no desea usar embeddings, ponga `EMBEDDINGS_PROVIDER=lexico` en `.env`; la recuperación usa solo BM25.

## Configuración

Edite `.env` (la plantilla es `.env.example`; no contiene claves reales).

| Variable | Uso |
|---|---|
| `LLM_PROVIDER` | Proveedor por defecto: `ollama`, `claude` u `openai` |
| `OLLAMA_MODEL` | Modelo local (por defecto `qwen2.5:3b`) |
| `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` | API de Claude |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | API con formato OpenAI. Ejemplos de Gemini, OpenAI y Groq en `.env.example` |
| `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL` | Embeddings locales o solo léxico |

Ejemplo con Gemini como API externa:

```
OPENAI_API_KEY=su_clave
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
OPENAI_MODEL=gemini-flash-lite-latest
```

Los cambios del `.env` se leen al iniciar la aplicación.

## Ejecución

**Windows:** `iniciar.bat`. **Linux / Mac:** `./iniciar.sh`. Abra <http://127.0.0.1:8000>.

En la pestaña **Codificar**:

1. Elija el método (con modelo de lenguaje o solo reglas) y el modelo (local o API). El punto de color indica si está disponible.
2. Cargue un ejemplo o pegue una nota.
3. Pulse **Sugerir códigos**. El panel muestra los pasos del pipeline con su duración.
4. En la tabla, cada código muestra la frase que lo justifica. Al pulsar una fila se resalta la frase en la nota. Las menciones negadas, descartadas, inciertas o familiares aparecen en **No codificadas**.

La pestaña **Evaluación** muestra las métricas guardadas en `data/resultados/resumen.json`.

Con el modelo local y sin GPU, una nota corta tarda entre 30 y 60 segundos si hay memoria libre. El modo **Solo reglas** responde en menos de un segundo.

## Evaluación

Los casos con códigos conocidos están en `data/casos/` (muestra de 40 casos del conjunto test de CodiEsp en `evaluacion.jsonl`). Para evaluar:

```bash
python scripts/evaluar.py --sistema reglas --conjunto test
python scripts/evaluar.py --sistema ollama --conjunto evaluacion --n 15
python scripts/evaluar.py --sistema openai --conjunto evaluacion --n 15
python scripts/resumen_evaluacion.py
```

`evaluar.py` guarda cada caso en `data/resultados/` y continúa donde quedó si se interrumpe. `resumen_evaluacion.py` calcula precisión, recall y F1, y arma el resumen que lee la interfaz.

Los conjuntos completos `train`, `dev` y `test` no se incluyen en el repositorio. Para usarlos (por ejemplo `--conjunto test`) o para regenerar todos los datos desde las fuentes públicas:

```bash
python scripts/descargar_datos.py
python scripts/preparar_datos.py
python scripts/ajustar_recuperacion.py   # recall@k de la recuperación (usa dev)
```

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Estructura

```
app/
  config.py            configuración (.env)
  main.py              API REST y servidor de la interfaz
  llm/                 proveedores: ollama.py, claude.py, openai_compat.py, embeddings.py
  pipeline/            preprocesamiento, negacion, catalogo, recuperacion, prompts, codificador
  evaluacion/          métricas y políticas de negación
  static/              interfaz (index.html, app.js, styles.css)
data/
  catalogo/            catálogo CIE-10 y expresiones del entrenamiento
  negex/               frases de negación
  casos/               casos de CodiEsp con sus códigos
  ejemplos/            notas ficticias de demostración
  indices/             embeddings de las categorías
  resultados/          resultados de la evaluación
scripts/               descargar_datos, preparar_datos, construir_indice, evaluar, resumen_evaluacion
tests/                 pruebas automáticas
```

## Datos y licencias

- Casos clínicos y códigos: corpus CodiEsp (CLEF eHealth 2020), CC BY 4.0. Miranda-Escalada et al. (2020). <https://doi.org/10.5281/zenodo.3837305>
- Catálogo CIE-10 (diagnósticos, versión 2018): lista de códigos de CodiEsp, CC BY 4.0. <https://doi.org/10.5281/zenodo.3706838>
- Negación: NegEx-MES, licencia MIT. <https://doi.org/10.5281/zenodo.2159627>

El proyecto no usa historias clínicas reales de pacientes. Las notas de `data/ejemplos/` son ficticias.

## Limitaciones

- Es un prototipo de apoyo: el codificador humano debe validar cada sugerencia.
- El modelo local de 3 mil millones de parámetros omite menciones y se equivoca más que un modelo grande.
- El corpus CodiEsp codifica también diagnósticos negados; para la evaluación se reportan ambos criterios.
- El catálogo se limita a códigos de hasta 6 caracteres; no cubre las extensiones de séptimo carácter.
