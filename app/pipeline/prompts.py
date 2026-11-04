"""Prompts y esquemas JSON de las dos llamadas al LLM.

Llamada 1 (extracción): menciones clínicas de la nota con su estado.
Llamada 2 (selección): código CIE-10 de cada mención, elegido entre los
candidatos que entrega la recuperación.
"""

ESTADOS = ["afirmado", "negado", "descartado", "incierto", "familiar"]

SISTEMA_EXTRACCION = """Eres un codificador clínico experto. Lees una nota clínica en español y extraes las menciones de enfermedades, diagnósticos, síntomas, signos y hallazgos patológicos que podrían recibir un código CIE-10.

Reglas:
- No extraigas medicamentos, procedimientos, pruebas ni resultados normales.
- "frase" debe copiarse EXACTAMENTE como aparece en la nota, lo más corta posible (el nombre de la enfermedad o síntoma, sin negaciones ni palabras de certeza).
- "estado" indica qué dice la nota sobre esa mención:
  afirmado: el paciente lo tiene o lo tuvo (incluye antecedentes personales).
  negado: la nota dice que no lo tiene (niega, sin, no presenta, ausencia de).
  descartado: se estudió y se excluyó.
  incierto: es posible, probable o solo una sospecha.
  familiar: afecta a un familiar, no al paciente.
- Si una enfermedad aparece varias veces, inclúyela una sola vez.
- Máximo 15 menciones.

Ejemplo:
Nota: "Paciente con HTA y fiebre desde hace 3 días. Niega tos. Se descarta neumonía. Madre con diabetes."
Respuesta: {"menciones":[{"frase":"HTA","estado":"afirmado"},{"frase":"fiebre","estado":"afirmado"},{"frase":"tos","estado":"negado"},{"frase":"neumonía","estado":"descartado"},{"frase":"diabetes","estado":"familiar"}]}"""

ESQUEMA_EXTRACCION = {
    "type": "object",
    "properties": {
        "menciones": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "properties": {
                    "frase": {"type": "string", "maxLength": 80},
                    "estado": {"type": "string", "enum": ESTADOS},
                },
                "required": ["frase", "estado"],
            },
        }
    },
    "required": ["menciones"],
}

SISTEMA_SELECCION = """Eres un codificador clínico experto en CIE-10-ES (diagnósticos). Para cada mención elige, entre los códigos candidatos, el que mejor la representa según la oración en que aparece.

Reglas:
- Responde solo con un código de la lista de candidatos de esa mención.
- Prefiere el código más específico que la oración permita; si no hay datos para precisar, elige el "no especificado".
- Si ningún candidato corresponde, responde NINGUNO.
- No inventes códigos."""

ESQUEMA_SELECCION = {
    "type": "object",
    "properties": {
        "decisiones": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "codigo": {"type": "string"},
                },
                "required": ["id", "codigo"],
            },
        }
    },
    "required": ["decisiones"],
}


def prompt_extraccion(nota: str) -> str:
    return f'Nota clínica:\n"""\n{nota}\n"""\n\nExtrae las menciones.'


def prompt_seleccion(bloques: list[dict]) -> str:
    """bloques: [{id, frase, oracion, candidatos:[(codigo, descripcion)]}]"""
    partes = []
    for b in bloques:
        lineas = "\n".join(f"  - {c} {d}" for c, d in b["candidatos"])
        partes.append(f'Mención {b["id"]}: "{b["frase"]}"\nOración: {b["oracion"]}\nCandidatos:\n{lineas}')
    return "\n\n".join(partes) + "\n\nElige el código de cada mención."
