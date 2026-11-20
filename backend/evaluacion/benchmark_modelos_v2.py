"""
Benchmark de modelos generativos para COODIBOT (OpenAI y Gemini).

Evalúa cada modelo sobre las 50 preguntas de dataset_capturado_50.json, usando los
mismos fragmentos recuperados y el mismo prompt de sistema de rag_service.py, de modo
que la única variable entre corridas es el modelo generador.

Métricas por modelo: latencia, tokens, costo por consulta, cumplimiento de formato,
abstención en preguntas trampa y RAGAS (Faithfulness, Answer Relevancy, Factual Correctness).

Uso (desde backend/):
  python evaluacion/benchmark_modelos_v2.py --listar     # modelos disponibles por proveedor
  python evaluacion/benchmark_modelos_v2.py --generar    # genera respuestas (reanudable)
  python evaluacion/benchmark_modelos_v2.py --evaluar    # métricas RAGAS
  python evaluacion/benchmark_modelos_v2.py --resumen    # tabla comparativa
  --muestra N limita la ejecución a las primeras N preguntas.

Variables de entorno: OPENAI_API_KEY, GEMINI_API_KEY.
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_EVALUACION = os.path.dirname(os.path.abspath(__file__))
RUTA_BACKEND = os.path.abspath(os.path.join(RUTA_EVALUACION, ".."))
RUTA_RAIZ = os.path.abspath(os.path.join(RUTA_BACKEND, ".."))
for ruta_env in (os.path.join(RUTA_BACKEND, ".env"), os.path.join(RUTA_RAIZ, ".env")):
    if os.path.exists(ruta_env):
        load_dotenv(ruta_env)

ARCHIVO_DATASET = os.path.join(RUTA_EVALUACION, "dataset_capturado_50.json")
CSV_RESPUESTAS = os.path.join(RUTA_EVALUACION, "benchmark_v2_respuestas.csv")
CARPETA_RAGAS = os.path.join(RUTA_EVALUACION, "benchmark_v2_ragas")
CSV_RESUMEN = os.path.join(RUTA_EVALUACION, "benchmark_v2_resumen.csv")

PROVEEDORES = {
    "openai": {"base_url": None, "env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "env": "GEMINI_API_KEY"},
}

# Precios en USD por 1M de tokens (entrada / salida), referencia nov-2026.
MODELOS = [
    # --- OpenAI ---
    {"nombre": "gpt-4o-mini (actual)", "proveedor": "openai", "modelo": "gpt-4o-mini",
     "precio_in": 0.15, "precio_out": 0.60, "extra": {}, "activo": True},
    {"nombre": "gpt-4.1-mini", "proveedor": "openai", "modelo": "gpt-4.1-mini",
     "precio_in": 0.40, "precio_out": 1.60, "extra": {}, "activo": True},
    {"nombre": "gpt-5.4-mini", "proveedor": "openai", "modelo": "gpt-5.4-mini",
     "precio_in": 0.75, "precio_out": 4.50, "extra": {"reasoning_effort": "none"}, "activo": True},
    {"nombre": "gpt-4o", "proveedor": "openai", "modelo": "gpt-4o",
     "precio_in": 2.50, "precio_out": 10.00, "extra": {}, "activo": False},
    # --- Google Gemini ---
    # gemini-2.5-flash-lite no está disponible para cuentas nuevas
    {"nombre": "gemini-3.5-flash-lite", "proveedor": "gemini", "modelo": "gemini-3.5-flash-lite",
     "precio_in": 0.30, "precio_out": 2.50, "extra": {"reasoning_effort": "low"}, "activo": True},
    {"nombre": "gemini-3.1-flash-lite", "proveedor": "gemini", "modelo": "gemini-3.1-flash-lite",
     "precio_in": 0.25, "precio_out": 1.50, "extra": {"reasoning_effort": "low"}, "activo": True},
    {"nombre": "gemini-3.5-flash", "proveedor": "gemini", "modelo": "gemini-3.5-flash",
     "precio_in": 1.50, "precio_out": 9.00, "extra": {"reasoning_effort": "low"}, "activo": True},
    {"nombre": "gemini-3-flash-preview", "proveedor": "gemini", "modelo": "gemini-3-flash-preview",
     "precio_in": 0.50, "precio_out": 3.00, "extra": {"reasoning_effort": "low"}, "activo": False},
]

TEMPERATURA = 0.1                          # igual a rag_service.py
CONCURRENCIA = {"openai": 2, "gemini": 1}  # solicitudes simultáneas por proveedor
MAX_REINTENTOS = 6

# ---------------------------------------------------------------------------
# Prompt (replica el de rag_service.py)
# ---------------------------------------------------------------------------
def construir_prompt(item: dict) -> tuple[str, list[str]]:
    """Retorna el prompt de sistema y los contextos entregados al modelo (fragmentos + OA)."""
    fragmentos = [c for c in item["contexts_rag"]
                  if not c.startswith("Sin fragmentos recuperados")]
    contexto = "".join(f + "\n\n---\n\n" for f in fragmentos)

    oas = item.get("oas_vinculados") or []
    lineas_oa = []
    for oa in oas:
        codigo = oa["codigo"] if isinstance(oa, dict) else str(oa)
        desc = oa.get("descripcion", "") if isinstance(oa, dict) else ""
        if desc:
            linea = f"[INFO PEDAGÓGICA OCULTA] El objetivo {codigo} trata sobre: {desc}"
            lineas_oa.append(linea)
            contexto += "\n" + linea + "\n"

    if oas:
        codigos = [oa["codigo"] if isinstance(oa, dict) else str(oa) for oa in oas]
        oa_txt = "\n           ".join(f"- {c}" for c in codigos)
    else:
        oa_txt = "Ninguno (Consulta puramente técnica)"

    prompt = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.

        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        3. Las OAs solo existen para los cursos: primero, segundo, tercero, cuarto, quinto y sexto básico.
        4. Si se te pregunta por algun curso fuera del rango de entre primero y sexto basico, puedes sugerir OAs de cursos mas bajos pero dejando en claro que no es el curso solicitado. No inventes OAs que no existan en el catálogo oficial.
        5. Tus respuestas no deben asumir edad de los estudiantes. Solo enfocate en los cursos.
        6. Antes de aplicar esta regla, revisa con cuidado si el CONTEXTO RECUPERADO ya incluye información específica sobre el componente, sensor o funcionalidad de la consulta. Si el contexto SÍ lo menciona, ignora esta regla y responde normalmente usando esa información (regla 1). Solo si, después de revisar todo el contexto, este NO contiene ninguna mención al componente consultado, distingue dos casos: (a) si el contexto sugiere que ese componente simplemente no es parte del ecosistema COODI, responde que no tienes información al respecto; (b) si es un componente plausible dentro de la línea de robótica educativa pero no aparece en ningún fragmento del contexto, indica explícitamente que podría tratarse de algo planificado pero no implementado aún, en vez de inventar una respuesta.
        7. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
           - Concepto Clave: (Definición breve)
           - Pasos: (Instrucciones numeradas con verbos imperativos)
           - OA Vinculado:
           {oa_txt}
           - Verificación: (Cómo comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto}
        """
    return prompt, (fragmentos + lineas_oa) or ["(sin contexto)"]


# ---------------------------------------------------------------------------
# Métricas de formato y abstención
# ---------------------------------------------------------------------------
ETIQUETAS = ["concepto clave", "pasos", "oa vinculado", "verificación"]
PATRON_ABSTENCION = re.compile(
    r"no tengo informaci[oó]n|no dispongo de informaci[oó]n|planificad[oa]|"
    r"no (est[aá]|se encuentra) implementad|no implementad|no forma parte", re.I)


def analizar(texto: str, nota: str) -> dict:
    limpio = texto.replace("*", "").lower()
    cumple = all(f"{e}:" in limpio for e in ETIQUETAS)
    abstiene = bool(PATRON_ABSTENCION.search(texto))
    es_trampa = "trampa" in (nota or "").lower()
    palabras = len(texto.split())
    return {
        "cumple_4_partes": cumple,
        "se_abstiene": abstiene,
        "formato_valido": cumple or abstiene,
        "es_trampa": es_trampa,
        "trampa_ok": abstiene if es_trampa else None,
        "palabras": palabras,
        "supera_100_palabras": palabras > 100,
    }


# ---------------------------------------------------------------------------
# Clientes y llamadas
# ---------------------------------------------------------------------------
def crear_clientes() -> dict:
    clientes = {}
    for nombre, cfg in PROVEEDORES.items():
        llave = os.getenv(cfg["env"])
        if llave:
            clientes[nombre] = AsyncOpenAI(api_key=llave, base_url=cfg["base_url"]) \
                if cfg["base_url"] else AsyncOpenAI(api_key=llave)
        else:
            print(f"[omitido] {nombre}: {cfg['env']} no definida")
    return clientes


async def llamar(cliente, cfg: dict, mensajes: list) -> tuple:
    """Solicitud con reintentos. Omite temperature/reasoning_effort si el modelo no los soporta."""
    params = {"model": cfg["modelo"], "messages": mensajes,
              "temperature": TEMPERATURA, **cfg["extra"]}
    quitados = []
    espera = 5
    for intento in range(MAX_REINTENTOS):
        try:
            t0 = time.perf_counter()
            resp = await cliente.chat.completions.create(**params)
            return resp, time.perf_counter() - t0, quitados
        except Exception as e:
            msg = str(e).lower()
            quitado = False
            for clave in ("temperature", "reasoning_effort"):
                if clave in params and clave in msg and ("unsupported" in msg or "not supported" in msg
                                                         or "invalid" in msg or "unknown" in msg
                                                         or "400" in msg):
                    params.pop(clave)
                    quitados.append(clave)
                    quitado = True
            if quitado:
                continue
            if any(s in msg for s in ("429", "rate limit", "quota", "resource_exhausted", "500", "503", "overloaded", "timeout")):
                print(f"   [reintento] {cfg['nombre']}: {espera}s")
                await asyncio.sleep(espera)
                espera = min(espera * 2, 90)
                continue
            raise
    raise RuntimeError(f"{cfg['nombre']}: reintentos agotados")


def cargar_respuestas_ok() -> pd.DataFrame:
    """Respuestas válidas, una por (modelo, pregunta)."""
    df = pd.read_csv(CSV_RESPUESTAS)
    df["id"] = df["id"].astype(str)
    df = df[(df["error"].fillna("") == "") & (df["respuesta"].fillna("").str.len() > 0)]
    return df.drop_duplicates(subset=["modelo", "id"], keep="last")


async def generar(muestra: int | None):
    with open(ARCHIVO_DATASET, encoding="utf-8") as f:
        dataset = json.load(f)
    if muestra:
        dataset = dataset[:muestra]

    clientes = crear_clientes()
    hechos = set()
    if os.path.exists(CSV_RESPUESTAS):
        previo = cargar_respuestas_ok()
        # Las filas con error se descartan para reintentarlas
        previo.to_csv(CSV_RESPUESTAS, index=False, encoding="utf-8-sig")
        hechos = set(zip(previo["modelo"], previo["id"].astype(str)))
        print(f"Respuestas existentes: {len(hechos)}")

    semaforos = {p: asyncio.Semaphore(CONCURRENCIA.get(p, 1)) for p in clientes}
    lock = asyncio.Lock()

    async def una(cfg, item):
        prompt, _ = construir_prompt(item)
        mensajes = [{"role": "system", "content": prompt},
                    {"role": "user", "content": item["question"]}]
        async with semaforos[cfg["proveedor"]]:
            try:
                resp, lat, quitados = await llamar(clientes[cfg["proveedor"]], cfg, mensajes)
                texto = (resp.choices[0].message.content or "").strip()
                uso = resp.usage
                t_in = getattr(uso, "prompt_tokens", 0) or 0
                t_out = getattr(uso, "completion_tokens", 0) or 0
                det = getattr(uso, "completion_tokens_details", None)
                t_razon = (getattr(det, "reasoning_tokens", 0) or 0) if det else 0
                error = ""
            except Exception as e:
                texto, lat, quitados, t_in, t_out, t_razon = "", None, [], 0, 0, 0
                error = str(e)[:300]
                print(f"   [ERROR] {cfg['nombre']} {item['id']}: {error}")

        fila = {
            "modelo": cfg["nombre"], "proveedor": cfg["proveedor"], "id_modelo": cfg["modelo"],
            "id": item["id"], "tipo": item["tipo"], "categoria": item["categoria"],
            "latencia_seg": None if lat is None else round(lat, 3),
            "tokens_in": t_in, "tokens_out": t_out, "tokens_razonamiento": t_razon,
            "costo_usd": round(t_in * cfg["precio_in"] / 1e6 + t_out * cfg["precio_out"] / 1e6, 7),
            "parametros_quitados": ",".join(quitados), "error": error,
            "respuesta": texto,
            **analizar(texto, item.get("nota", "")),
        }
        async with lock:
            pd.DataFrame([fila]).to_csv(CSV_RESPUESTAS, mode="a", index=False,
                                        header=not os.path.exists(CSV_RESPUESTAS),
                                        encoding="utf-8-sig")
        print(f"   {cfg['nombre']:<24} {item['id']}  {fila['latencia_seg']} s")

    tareas = []
    for cfg in MODELOS:
        if not cfg["activo"] or cfg["proveedor"] not in clientes:
            continue
        for item in dataset:
            if (cfg["nombre"], str(item["id"])) not in hechos:
                tareas.append(una(cfg, item))
    print(f"Respuestas pendientes: {len(tareas)}")
    await asyncio.gather(*tareas)
    print(f"\nRespuestas guardadas en {CSV_RESPUESTAS}")


async def listar():
    clientes = crear_clientes()
    for prov, cliente in clientes.items():
        try:
            ids = sorted(m.id for m in (await cliente.models.list()).data)
        except Exception as e:
            print(f"\n{prov}: error al listar modelos ({e})")
            continue
        print(f"\n=== {prov}: {len(ids)} modelos disponibles ===")
        configurados = {c["modelo"] for c in MODELOS if c["proveedor"] == prov}
        for c in sorted(configurados):
            ok = any(i == c or i.endswith("/" + c) for i in ids)
            print(f"  {'OK   ' if ok else 'N/D  '} {c}")
        candidatos = [i for i in ids if any(k in i for k in ("mini", "nano", "flash", "lite"))]
        print("  Otros modelos livianos:", ", ".join(candidatos[:30]))


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
def evaluar():
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, ResponseRelevancy
    metricas = [Faithfulness(), ResponseRelevancy()]
    try:
        from ragas.metrics import FactualCorrectness
        metricas.append(FactualCorrectness())
    except ImportError:
        print("[omitido] FactualCorrectness no disponible en esta versión de ragas")

    api_key = os.getenv("OPENAI_API_KEY")
    juez = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    config = RunConfig(max_workers=4, timeout=180)

    with open(ARCHIVO_DATASET, encoding="utf-8") as f:
        por_id = {str(d["id"]): d for d in json.load(f)}
    resp = cargar_respuestas_ok()
    os.makedirs(CARPETA_RAGAS, exist_ok=True)

    for nombre, grupo in resp.groupby("modelo"):
        salida = os.path.join(CARPETA_RAGAS, re.sub(r"[^\w.-]+", "_", nombre) + ".csv")
        if os.path.exists(salida):
            print(f"[existente] {nombre}")
            continue
        print(f"\nRAGAS: {nombre} (n={len(grupo)})")
        contextos = [construir_prompt(por_id[i])[1] for i in grupo["id"]]
        ds = Dataset.from_dict({
            "question": [por_id[i]["question"] for i in grupo["id"]],
            "answer": grupo["respuesta"].tolist(),
            "contexts": contextos,
            "ground_truth": [por_id[i]["ground_truth"] for i in grupo["id"]],
        })
        res = evaluate(dataset=ds, metrics=metricas, llm=juez, embeddings=emb, run_config=config)
        df = res.to_pandas()
        renombres = {}
        for col in df.columns:
            if col.startswith("faithfulness"):
                renombres[col] = "Faithfulness"
            elif col.startswith(("answer_relevancy", "response_relevancy")):
                renombres[col] = "Answer Relevancy"
            elif col.startswith("factual_correctness"):
                renombres[col] = "Factual Correctness"
        df = df.rename(columns=renombres)
        df["modelo"] = nombre
        df["id"] = grupo["id"].tolist()
        df.to_csv(salida, index=False, encoding="utf-8-sig")
        print(f"Guardado en {salida}")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def p90(serie):
    s = sorted(serie.dropna())
    return s[max(0, int(round(0.9 * len(s))) - 1)] if s else None


def resumen():
    resp = cargar_respuestas_ok()
    filas = []
    for nombre, ok in resp.groupby("modelo", sort=False):
        trampas = ok[ok["es_trampa"] == True]
        fila = {
            "Modelo": nombre,
            "N": len(ok),
            "Latencia media (s)": round(ok["latencia_seg"].mean(), 2),
            "Latencia mediana (s)": round(ok["latencia_seg"].median(), 2),
            "Latencia p90 (s)": round(p90(ok["latencia_seg"]), 2),
            "Tokens entrada": round(ok["tokens_in"].mean()),
            "Tokens salida": round(ok["tokens_out"].mean()),
            "Costo por consulta (USD)": round(ok["costo_usd"].mean(), 6),
            "Costo por 1000 consultas (USD)": round(ok["costo_usd"].mean() * 1000, 3),
            "Palabras": round(ok["palabras"].mean(), 1),
            "% > 100 palabras": round(100 * ok["supera_100_palabras"].mean(), 1),
            "% formato válido": round(100 * ok["formato_valido"].mean(), 1),
            "Trampas abstenidas": f"{int(trampas['trampa_ok'].sum())}/{len(trampas)}",
            "Parámetros quitados": ",".join(sorted({p for p in ok["parametros_quitados"].fillna("") if p})),
        }
        archivo = os.path.join(CARPETA_RAGAS, re.sub(r"[^\w.-]+", "_", nombre) + ".csv")
        if os.path.exists(archivo):
            r = pd.read_csv(archivo)
            for m in ("Faithfulness", "Answer Relevancy", "Factual Correctness"):
                if m in r.columns:
                    fila[m] = round(r[m].mean(), 3)
        filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(CSV_RESUMEN, index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string(index=False))
    print(f"\nGuardado en {CSV_RESUMEN}")
    print("Obs.: la abstención se detecta por palabras clave; requiere validación manual.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--generar", action="store_true")
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--muestra", type=int, default=None)
    args = ap.parse_args()
    if not any((args.listar, args.generar, args.evaluar, args.resumen)):
        ap.print_help()
        sys.exit(0)
    if args.listar:
        asyncio.run(listar())
    if args.generar:
        asyncio.run(generar(args.muestra))
    if args.evaluar:
        evaluar()
    if args.resumen:
        resumen()