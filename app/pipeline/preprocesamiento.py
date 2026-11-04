"""Preprocesamiento: oraciones, abreviaturas, normalización y tokens."""
from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_stemmer = snowballstemmer.stemmer("spanish")

PALABRAS_VACIAS = {
    "a", "al", "algo", "ante", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellos", "en", "entre", "era", "es", "esta", "este", "esto",
    "fue", "ha", "han", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "muy",
    "o", "otra", "otro", "para", "pero", "por", "que", "se", "ser", "si", "sin", "sobre", "su",
    "sus", "tambien", "un", "una", "uno", "y", "ya",
}

# Abreviaturas clínicas frecuentes en notas en español. Se desarrollan solo
# para buscar en el catálogo; el texto original no se modifica.
ABREVIATURAS = {
    "hta": "hipertensión arterial",
    "dm": "diabetes mellitus",
    "dm1": "diabetes mellitus tipo 1",
    "dm2": "diabetes mellitus tipo 2",
    "iam": "infarto agudo de miocardio",
    "epoc": "enfermedad pulmonar obstructiva crónica",
    "acv": "accidente cerebrovascular",
    "avc": "accidente cerebrovascular",
    "icc": "insuficiencia cardiaca congestiva",
    "irc": "insuficiencia renal crónica",
    "tep": "tromboembolismo pulmonar",
    "tvp": "trombosis venosa profunda",
    "vih": "virus de inmunodeficiencia humana",
    "itu": "infección del tracto urinario",
    "ivu": "infección de vías urinarias",
    "eda": "enfermedad diarreica aguda",
    "hbp": "hiperplasia benigna de próstata",
    "fa": "fibrilación auricular",
    "erge": "enfermedad por reflujo gastroesofágico",
    "rge": "reflujo gastroesofágico",
    "tce": "traumatismo craneoencefálico",
    "sdra": "síndrome de dificultad respiratoria agudo",
    "hda": "hemorragia digestiva alta",
    "hdb": "hemorragia digestiva baja",
    "tbc": "tuberculosis",
    "tb": "tuberculosis",
    "lla": "leucemia linfoblástica aguda",
    "lma": "leucemia mieloide aguda",
    "ica": "insuficiencia cardiaca",
    "ic": "insuficiencia cardiaca",
    "ira": "infección respiratoria aguda",
    "nac": "neumonía adquirida en la comunidad",
    "dlp": "dislipidemia",
    "abd": "abdominal",
    "ca": "cáncer",
    "sca": "síndrome coronario agudo",
    "ecv": "enfermedad cerebrovascular",
    "ait": "accidente isquémico transitorio",
    "pcr": "proteína C reactiva",
    "lcr": "líquido cefalorraquídeo",
}

_ESTADO_PUNTO = re.compile(r"[.;!?]\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9(\"«])")
_ABREV_CON_PUNTO = {"dr", "dra", "sr", "sra", "aprox", "pag", "fig", "vs", "etc", "tel", "no", "num"}


def plegar(texto: str) -> str:
    """Minúsculas y sin diacríticos, carácter a carácter (misma longitud)."""
    salida = []
    for c in texto:
        base = unicodedata.normalize("NFD", c)[:1] or c
        bajo = base.lower()
        salida.append(bajo if len(bajo) == 1 else base)
    return "".join(salida)


def oraciones(texto: str) -> list[tuple[int, int]]:
    """Posiciones (inicio, fin) de cada oración. Corta en saltos de línea y
    en . ; ! ? seguidos de mayúscula, salvo abreviaturas como 'Dr.'."""
    cortes = [0]
    for m in re.finditer(r"\n+", texto):
        cortes.append(m.end())
    for m in _ESTADO_PUNTO.finditer(texto):
        previo = re.search(r"([A-Za-zÁÉÍÓÚáéíóúñÑ]+)$", texto[max(0, m.start() - 8) : m.start()])
        if m.group()[0] == "." and previo and plegar(previo.group(1)) in _ABREV_CON_PUNTO:
            continue
        cortes.append(m.end())
    cortes = sorted(set(cortes)) + [len(texto)]
    spans = []
    for ini, fin in zip(cortes, cortes[1:]):
        trozo = texto[ini:fin]
        izq = len(trozo) - len(trozo.lstrip())
        der = len(trozo.rstrip())
        if der - izq > 0:
            spans.append((ini + izq, ini + der))
    return spans


def expandir_abreviaturas(frase: str) -> str:
    """Desarrolla abreviaturas conocidas dentro de una frase."""

    def sustituir(m: re.Match) -> str:
        clave = plegar(m.group(0))
        if clave == "ca" and not frase[m.end() :].lstrip().lower().startswith("de "):
            return m.group(0)  # 'ca' solo es cáncer en 'ca de ...'
        return ABREVIATURAS.get(clave, m.group(0))

    return re.sub(r"\b[A-Za-zÁÉÍÓÚáéíóúñÑ]{1,5}\d?\b", sustituir, frase)


def tokens_busqueda(frase: str) -> list[str]:
    """Tokens normalizados para el índice léxico: sin tildes, sin palabras vacías, con stemming."""
    palabras = re.findall(r"[a-z0-9]+", plegar(frase))
    return [_stemmer.stemWord(p) for p in palabras if p not in PALABRAS_VACIAS and len(p) > 1 or p.isdigit()]


def normalizar_frase(frase: str) -> str:
    """Forma canónica de una frase para comparar de forma exacta."""
    return " ".join(re.findall(r"[a-z0-9]+", plegar(frase)))
