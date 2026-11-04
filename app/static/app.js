"use strict";

const $ = (id) => document.getElementById(id);
const estado = { ejemplos: [], api: null, resultado: null, evaluacion: null };

const PASOS = [
  ["preprocesamiento", "Preprocesamiento", 2],
  ["extraccion", "Extracción de menciones", 3],
  ["negacion", "Detección de negación", 3],
  ["recuperacion", "Recuperación de candidatos", 3],
  ["seleccion", "Selección del código", 3],
  ["consolidacion", "Consolidación", 3],
];

function esc(t) {
  return String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------ pestañas */
document.querySelectorAll(".pestanas button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".pestanas button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $("vista-codificar").hidden = b.dataset.vista !== "codificar";
    $("vista-evaluacion").hidden = b.dataset.vista !== "evaluacion";
    if (b.dataset.vista === "evaluacion") cargarEvaluacion();
  })
);

/* ------------------------------------------------------ estado y ajustes */
async function cargarEstado() {
  try {
    estado.api = await (await fetch("/api/estado")).json();
    if (!$("proveedor").dataset.fijado) $("proveedor").value = estado.api.predeterminado;
  } catch {
    estado.api = null;
    setTimeout(cargarEstado, 3000);
  }
  pintarEstadoProveedor();
}

function pintarEstadoProveedor() {
  const caja = $("estado-proveedor");
  const modoReglas = $("modo").value === "reglas";
  $("grupo-proveedor").style.opacity = modoReglas ? ".45" : "1";
  if (!estado.api) {
    caja.className = "estado mal";
    $("estado-texto").textContent = "Sin conexión con el servidor";
    return;
  }
  const p = estado.api.proveedores[$("proveedor").value];
  caja.className = "estado " + (p.disponible ? "ok" : "mal");
  $("estado-texto").textContent = `${p.modelo} · ${p.mensaje}`;
}
$("proveedor").addEventListener("change", () => { $("proveedor").dataset.fijado = "1"; pintarEstadoProveedor(); });
$("modo").addEventListener("change", pintarEstadoProveedor);

/* ------------------------------------------------------------ ejemplos */
async function cargarEjemplos() {
  estado.ejemplos = await (await fetch("/api/ejemplos")).json();
  const sel = $("ejemplos");
  estado.ejemplos.forEach((e, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = (e.origen === "ficticio" ? "Ficticio · " : "CodiEsp · ") + e.titulo;
    sel.appendChild(o);
  });
}
$("ejemplos").addEventListener("change", (ev) => {
  if (ev.target.value === "") return;
  modoEdicion();
  $("nota").value = estado.ejemplos[ev.target.value].texto;
  actualizarConteo();
  limpiarResultados();
});

function actualizarConteo() {
  const n = $("nota").value.trim().split(/\s+/).filter(Boolean).length;
  $("conteo").textContent = `${n} palabras`;
}
$("nota").addEventListener("input", actualizarConteo);

/* ---------------------------------------------------------- análisis */
function modoEdicion() {
  $("nota").hidden = false;
  $("nota-anotada").hidden = true;
  $("btn-editar").hidden = true;
  $("leyenda").hidden = true;
}
$("btn-editar").addEventListener("click", modoEdicion);

function limpiarResultados() {
  $("resultados").hidden = true;
  $("mensaje").hidden = true;
  $("pipeline").innerHTML = "";
}

function pintarPipeline(definidos, pasos) {
  $("pipeline").innerHTML = definidos
    .map(([id, nombre, fase]) => {
      const p = pasos[id];
      const cls = p ? (p.estado === "listo" ? "listo" : "activo") : "";
      const detalle = p && p.estado === "listo" ? esc(p.detalle) : p ? "en curso" : "pendiente";
      return `<li class="${cls}"><b>${esc(nombre)}</b><span>Fase ${fase} · ${detalle}</span></li>`;
    })
    .join("");
}

function mostrarMensaje(texto, info = false) {
  const m = $("mensaje");
  m.textContent = texto;
  m.className = "mensaje" + (info ? " info" : "");
  m.hidden = !texto;
}

$("btn-codificar").addEventListener("click", async () => {
  const nota = $("nota").value.trim();
  if (nota.length < 10) return mostrarMensaje("Escriba o cargue una nota clínica de al menos 10 caracteres.");
  const boton = $("btn-codificar");
  boton.disabled = true;
  limpiarResultados();
  const negados = $("op-negados").checked;
  const cuerpo = {
    nota,
    modo: $("modo").value,
    proveedor: $("proveedor").value,
    politica: { usar_negex: negados, usar_estado_llm: negados, excluir_inciertos: $("op-inciertos").checked, puntaje_minimo: parseFloat($("confianza").value) },
  };
  pintarPipeline(PASOS, {});
  try {
    const r = await fetch("/api/codificar", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });
    if (!r.ok) throw new Error("El servidor rechazó la nota (revise su longitud).");
    const { id } = await r.json();
    if (cuerpo.modo === "llm" && cuerpo.proveedor === "ollama")
      mostrarMensaje("Con el modelo local el análisis tarda varios minutos en equipos sin GPU.", true);
    await seguir(id, nota);
  } catch (e) {
    mostrarMensaje(e.message || "No se pudo completar el análisis.");
  } finally {
    boton.disabled = false;
  }
});

async function seguir(id, nota) {
  for (;;) {
    const t = await (await fetch(`/api/trabajos/${id}`)).json();
    pintarPipeline(PASOS, t.pasos || {});
    if (t.estado === "listo") {
      mostrarMensaje("");
      estado.resultado = t.resultado;
      pintarResultados(nota, t.resultado, t.segundos);
      return;
    }
    if (t.estado === "error") return mostrarMensaje(t.error);
    await new Promise((ok) => setTimeout(ok, 1000));
  }
}

/* --------------------------------------------------------- resultados */
function claseMencion(m, r, motivos, codificadas) {
  if (codificadas.has(m.id)) return "m-codificada";
  const motivo = motivos.get(m.id);
  if (motivo === "negado" || motivo === "descartado") return "m-negado";
  if (motivo === "incierto") return "m-incierto";
  if (motivo === "familiar") return "m-familiar";
  return "m-sincodigo";
}

function pintarResultados(nota, r, segundos) {
  const codificadas = new Set(r.codigos.flatMap((c) => c.evidencias.map((e) => e.mencion)));
  const motivos = new Map(r.descartados.map((d) => [d.mencion, d.motivo]));

  // nota anotada: se recorren las menciones por posición, sin solapamientos
  const ordenadas = r.menciones.filter((m) => m.ini >= 0).sort((a, b) => a.ini - b.ini || b.fin - a.fin);
  let html = "", cursor = 0;
  for (const m of ordenadas) {
    if (m.ini < cursor) continue;
    html += esc(nota.slice(cursor, m.ini));
    html += `<span class="m ${claseMencion(m, r, motivos, codificadas)}" data-m="${m.id}">${esc(nota.slice(m.ini, m.fin))}</span>`;
    cursor = m.fin;
  }
  html += esc(nota.slice(cursor));
  const caja = $("nota-anotada");
  caja.innerHTML = html;
  caja.hidden = false;
  $("nota").hidden = true;
  $("btn-editar").hidden = false;
  $("leyenda").hidden = false;

  // tabla de códigos
  const cuerpo = $("tabla-codigos").querySelector("tbody");
  cuerpo.innerHTML = r.codigos
    .map((c) => {
      const e = c.evidencias[0];
      const mas = c.evidencias.length > 1 ? ` (+${c.evidencias.length - 1} más)` : "";
      const oracion = esc(e.oracion.replace(/\s+/g, " ").slice(0, 170));
      return `<tr data-c="${esc(c.codigo)}" data-m="${c.evidencias.map((x) => x.mencion).join(",")}">
        <td class="codigo">${esc(c.codigo)}</td>
        <td>${esc(c.descripcion)}</td>
        <td class="evidencia"><q>${esc(e.frase)}</q>${mas}<small>${oracion}</small></td>
        <td><span class="chip chip-${c.confianza}">${c.confianza}</span><span class="fuente">${esc(c.fuente)}</span></td></tr>`;
    })
    .join("");
  $("vacio-codigos").hidden = r.codigos.length > 0;
  $("tabla-codigos").hidden = r.codigos.length === 0;
  $("contador-codigos").textContent = `(${r.codigos.length})`;
  const modelo = r.modo === "llm" ? `${r.modelo}${r.semantica ? " + embeddings" : ""}` : r.modelo;
  $("resumen-corrida").textContent = `${modelo} · ${segundos} s`;

  // no codificadas
  $("bloque-descartados").hidden = r.descartados.length === 0;
  $("contador-desc").textContent = `(${r.descartados.length})`;
  $("lista-desc").innerHTML = r.descartados
    .map(
      (d) => `<li><span class="motivo motivo-${esc(d.motivo)}">${esc(d.motivo)}</span>
        <span><b>${esc(d.frase)}</b> <span class="discreto">→ ${esc(d.codigo)} ${esc(d.descripcion)}${d.disparador ? ` · indicio: «${esc(d.disparador)}»` : ""}</span></span></li>`
    )
    .join("");
  $("resultados").hidden = false;

  cuerpo.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => {
      cuerpo.querySelectorAll("tr").forEach((x) => x.classList.remove("activa"));
      caja.querySelectorAll(".m.activa").forEach((x) => x.classList.remove("activa"));
      tr.classList.add("activa");
      let primero = null;
      tr.dataset.m.split(",").forEach((id) => {
        const el = caja.querySelector(`.m[data-m="${id}"]`);
        if (el) { el.classList.add("activa"); primero = primero || el; }
      });
      if (primero) primero.scrollIntoView({ block: "center", behavior: "smooth" });
    })
  );
}

/* --------------------------------------------------------- evaluación */
const VARIANTES = {
  sin_negacion: "Sin filtro de negación",
  solo_reglas: "Filtro con reglas (NegEx)",
  solo_llm: "Filtro con estado del modelo",
  reglas_y_llm: "Reglas y modelo",
};

async function cargarEvaluacion() {
  if (!estado.evaluacion) estado.evaluacion = await (await fetch("/api/evaluacion")).json();
  const ev = estado.evaluacion;
  if (!ev.conjuntos || !ev.conjuntos.length) {
    $("evaluacion-contenido").innerHTML = '<p class="vacio">Aún no hay resultados. Ejecute scripts/evaluar.py y scripts/resumen_evaluacion.py.</p>';
    return;
  }
  const selC = $("sel-conjunto"), selV = $("sel-variante");
  if (!selC.options.length) {
    ev.conjuntos.forEach((c, i) => selC.add(new Option(`${c.titulo} (${c.casos} casos)`, i)));
    Object.entries(VARIANTES).forEach(([k, v]) => selV.add(new Option(v, k)));
    selV.value = "sin_negacion";
    selC.addEventListener("change", pintarEvaluacion);
    selV.addEventListener("change", pintarEvaluacion);
    $("sel-criterio").addEventListener("change", pintarEvaluacion);
  }
  pintarEvaluacion();
}

function barra(etiqueta, v) {
  return `<span>${etiqueta}</span><div class="barra-fondo"><div class="barra-valor" style="width:${(v * 100).toFixed(1)}%"></div></div><b>${v.toFixed(3)}</b>`;
}

function pintarEvaluacion() {
  const ev = estado.evaluacion;
  const conjunto = ev.conjuntos[$("sel-conjunto").value];
  const variante = $("sel-variante").value;
  const hayAjustado = conjunto.sistemas.some((s) => s.variantes_ajustado);
  $("sel-criterio").disabled = !hayAjustado;
  const ajustado = hayAjustado && $("sel-criterio").value === "ajustado";
  const tomar = (s) => ((ajustado ? s.variantes_ajustado : s.variantes) || s.variantes)[variante] || s.variantes.sin_negacion;
  $("evaluacion-nota").textContent = conjunto.nota || "";
  let html = '<table class="tabla tabla-eval"><thead><tr><th>Sistema</th><th>Precisión</th><th>Recall</th><th>F1</th><th>F1 por categoría</th><th>s/caso</th></tr></thead><tbody>';
  for (const s of conjunto.sistemas) {
    const m = tomar(s);
    html += `<tr><td>${esc(s.nombre)}<br><span class="discreto">${esc(s.modelo)}</span></td>
      <td>${m.exacto.precision.toFixed(3)}</td><td>${m.exacto.recall.toFixed(3)}</td><td><b>${m.exacto.f1.toFixed(3)}</b></td>
      <td>${m.categoria.f1.toFixed(3)}</td><td>${s.seg_caso}</td></tr>`;
  }
  html += "</tbody></table>";
  for (const s of conjunto.sistemas) {
    const m = tomar(s).exacto;
    html += `<div class="sistema"><h3>${esc(s.nombre)}</h3>
      <div class="meta"><span>${esc(s.modelo)}</span><span>${m.tp} aciertos · ${m.fp} sobrantes · ${m.fn} faltantes</span>
      ${s.tokens_entrada ? `<span>${Math.round(s.tokens_entrada)} tokens de entrada/caso · ${Math.round(s.tokens_salida)} de salida/caso</span>` : ""}</div>
      <div class="barras">${barra("Precisión", m.precision)}${barra("Recall", m.recall)}${barra("F1", m.f1)}</div></div>`;
  }
  const n = ev.negacion;
  if (n) {
    html += `<div class="sistema"><h3>Detección de negación (NegEx)</h3>
      <div class="meta"><span>${n.anotaciones} anotaciones revisadas a mano</span><span>${n.no_afirmadas_revision} negadas o inciertas</span></div>
      <div class="barras">${barra("Precisión", n.precision)}${barra("Recall", n.recall)}${barra("F1", n.f1)}</div></div>`;
  }
  $("evaluacion-contenido").innerHTML = html;
}

/* ------------------------------------------------------------- inicio */
cargarEstado();
cargarEjemplos();
actualizarConteo();
setInterval(cargarEstado, 15000);
