"""Pruebas del pipeline sin depender de Ollama ni de APIs externas."""
import unittest

from app import llm
from app.evaluacion.metricas import evaluar
from app.evaluacion.politicas import VARIANTES, codigos_con_politica
from app.llm.base import ProveedorLLM, extraer_json
from app.pipeline.codificador import Codificador, ubicar_frase
from app.pipeline.negacion import DetectorNegacion
from app.pipeline.preprocesamiento import expandir_abreviaturas, oraciones, plegar, tokens_busqueda


class ProveedorFalso(ProveedorLLM):
    """Responde con valores fijos: permite probar el pipeline sin modelo real."""

    nombre = "falso"
    modelo = "falso-1"

    def generar_json(self, sistema, prompt, schema, nombre):
        if nombre == "registrar_menciones":
            return {
                "menciones": [
                    {"frase": "HTA", "estado": "afirmado"},
                    {"frase": "fiebre", "estado": "negado"},
                    {"frase": "neumonía", "estado": "descartado"},
                    {"frase": "frase inventada", "estado": "afirmado"},
                ]
            }
        # selección: elige el primer candidato de cada mención
        import re

        decisiones = []
        for bloque in prompt.split("\n\n"):
            m = re.match(r'Mención (\d+):', bloque)
            c = re.search(r"  - (\S+) ", bloque)
            if m and c:
                decisiones.append({"id": int(m.group(1)), "codigo": c.group(1)})
        return {"decisiones": decisiones}

    def estado(self):
        return {"disponible": True, "modelo": self.modelo, "mensaje": "ok"}


class PruebasPreprocesamiento(unittest.TestCase):
    def test_plegar_conserva_longitud(self):
        texto = "Niño con ÁCIDO úrico"
        self.assertEqual(len(plegar(texto)), len(texto))
        self.assertEqual(plegar(texto), "nino con acido urico")

    def test_oraciones_no_corta_abreviaturas(self):
        t = "El Dr. Pérez indica 2,5 mg. Presenta HTA.\nSegunda línea."
        partes = [t[a:b] for a, b in oraciones(t)]
        self.assertEqual(partes, ["El Dr. Pérez indica 2,5 mg.", "Presenta HTA.", "Segunda línea."])

    def test_abreviaturas(self):
        self.assertIn("hipertensión arterial", expandir_abreviaturas("HTA"))
        self.assertEqual(expandir_abreviaturas("ca de mama"), "cáncer de mama")
        self.assertEqual(expandir_abreviaturas("Ca sérico"), "Ca sérico")

    def test_tokens_aplican_stemming_y_sin_vacias(self):
        self.assertEqual(tokens_busqueda("dolor de las rodillas"), tokens_busqueda("dolores rodilla"))


class PruebasNegacion(unittest.TestCase):
    def setUp(self):
        self.d = DetectorNegacion()

    def estado(self, oracion, frase):
        i = oracion.index(frase)
        return self.d.evaluar(oracion, i, i + len(frase)).estado

    def test_negado(self):
        self.assertEqual(self.estado("Paciente que niega fiebre ni vómitos.", "vómitos"), "negado")
        self.assertEqual(self.estado("Sin signos de insuficiencia cardiaca.", "insuficiencia cardiaca"), "negado")
        self.assertEqual(self.estado("Neumonía descartada.", "Neumonía"), "negado")

    def test_afirmado(self):
        self.assertEqual(self.estado("Paciente con fiebre y tos.", "tos"), "afirmado")

    def test_conjuncion_corta_el_alcance(self):
        self.assertEqual(self.estado("Sin fiebre pero con tos persistente.", "tos"), "afirmado")

    def test_incierto_y_pseudonegacion(self):
        self.assertEqual(self.estado("Probable cirrosis hepática.", "cirrosis"), "incierto")
        self.assertEqual(self.estado("No se descarta neumonía.", "neumonía"), "incierto")


class PruebasUtilidades(unittest.TestCase):
    def test_extraer_json_con_bloque_de_codigo(self):
        self.assertEqual(extraer_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extraer_json('Respuesta: {"a": 2} fin'), {"a": 2})

    def test_ubicar_frase_usa_apariciones_sucesivas(self):
        texto = "Fiebre alta. Luego fiebre."
        usadas = set()
        a = ubicar_frase(texto, "fiebre", usadas)
        usadas.add(a)
        b = ubicar_frase(texto, "fiebre", usadas)
        self.assertNotEqual(a, b)
        self.assertEqual(ubicar_frase(texto, "tos", usadas), (-1, -1))

    def test_metricas(self):
        reales = {"a": {"I10", "E11.9"}, "b": {"R50.9"}}
        pred = {"a": {"I10", "R05"}, "b": {"R50.9"}}
        m = evaluar(pred, reales)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 1, 1))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertEqual(evaluar({"a": {"E11.65"}}, {"a": {"E11.9"}}, "categoria")["f1"], 1.0)


class PruebasCodificador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        llm.PROVEEDORES["falso"] = ProveedorFalso
        cls.cod = Codificador(usar_semantica=False)
        cls.nota = "Paciente de 70 años con HTA. Niega fiebre. Se descarta neumonía."

    def test_modo_reglas_sin_llm(self):
        r = self.cod.codificar("Paciente con hipertensión arterial y sin fiebre.", modo="reglas")
        codigos = {c["codigo"] for c in r["codigos"]}
        self.assertIn("I10", codigos)
        self.assertNotIn("R50.9", codigos)
        self.assertTrue(any(d["codigo"] == "R50.9" and d["motivo"] == "negado" for d in r["descartados"]))

    def test_modo_llm_con_proveedor_falso(self):
        r = self.cod.codificar(self.nota, modo="llm", proveedor="falso")
        codigos = {c["codigo"] for c in r["codigos"]}
        self.assertIn("I10", codigos)
        self.assertEqual(len(r["menciones"]), 3)  # la frase inventada se descarta: no está en la nota
        motivos = {d["frase"]: d["motivo"] for d in r["descartados"]}
        self.assertEqual(motivos["fiebre"], "negado")
        self.assertIn(motivos["neumonía"], ("negado", "descartado"))  # ambos excluyen la mención
        self.assertEqual([p["id"] for p in r["pasos"]][0], "preprocesamiento")

    def test_politicas_de_negacion(self):
        r = self.cod.codificar(self.nota, modo="llm", proveedor="falso")
        sin = codigos_con_politica(r, VARIANTES["sin_negacion"])
        con = codigos_con_politica(r, VARIANTES["reglas_y_llm"])
        self.assertLess(len(con), len(sin))

    def test_confianza_minima_mueve_codigos_a_no_codificados(self):
        r = self.cod.codificar(self.nota, modo="llm", proveedor="falso", politica={"puntaje_minimo": 5.0})
        self.assertEqual(r["codigos"], [])
        self.assertTrue(any(d["motivo"] == "confianza baja" for d in r["descartados"]))

    def test_evidencia_apunta_al_texto(self):
        r = self.cod.codificar(self.nota, modo="llm", proveedor="falso")
        for c in r["codigos"]:
            for e in c["evidencias"]:
                self.assertEqual(self.nota[e["ini"] : e["fin"]], e["frase"])


if __name__ == "__main__":
    unittest.main()
