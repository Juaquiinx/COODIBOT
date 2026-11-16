import os
import sys
import json
import time
import asyncio
import importlib.util
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Detectar automáticamente la carpeta backend/ (un nivel arriba de evaluacion/)
RUTA_EVALUACION = os.path.dirname(os.path.abspath(__file__))
RUTA_BACKEND = os.path.abspath(os.path.join(RUTA_EVALUACION, ".."))

os.chdir(RUTA_BACKEND)
if RUTA_BACKEND not in sys.path:
    sys.path.insert(0, RUTA_BACKEND)

load_dotenv(os.path.join(RUTA_BACKEND, ".env"))


def cargar_modulo_desde_ruta(nombre_modulo: str, ruta_archivo: str):
    """Carga un archivo .py directamente por ruta absoluta sin depender de PYTHONPATH."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(
            f"No se encontró el archivo esperado en: {ruta_archivo}")
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta_archivo)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo


# Carga explícita de database.py y rag_service.py desde backend/
database_mod = cargar_modulo_desde_ruta(
    "database", os.path.join(RUTA_BACKEND, "database.py"))
rag_service_mod = cargar_modulo_desde_ruta(
    "rag_service", os.path.join(RUTA_BACKEND, "rag_service.py"))

_conectar = database_mod._conectar
procesar_rag = rag_service_mod.procesar_rag

cliente_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ARCHIVO_CSV = os.path.join(RUTA_EVALUACION, "preguntas_ragas.csv")
ARCHIVO_SALIDA_JSON = os.path.join(
    RUTA_EVALUACION, "dataset_capturado_50.json")


async def generar_respuesta_zero_shot(pregunta: str) -> str:
    """Línea base Zero-Shot: el mismo LLM (gpt-4o-mini) sin acceso a Pinecone ni catálogo de OAs."""
    prompt_base = (
        "Eres un asistente educativo que ayuda a docentes de educación básica en Chile "
        "a integrar el robot educativo COODI en sus clases. Responde de forma breve (máximo 100 palabras) "
        "y menciona qué Objetivo de Aprendizaje (OA) del currículum chileno se relaciona si corresponde."
    )
    respuesta = await cliente_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_base},
            {"role": "user", "content": pregunta}
        ],
        temperature=0.1
    )
    return respuesta.choices[0].message.content.strip()


def limpiar_sesiones_evaluacion():
    """Elimina de Turso los registros temporales creados durante el benchmark para no ensuciar historial_chat."""
    try:
        conn = _conectar()
        conn.execute(
            "DELETE FROM historial_chat WHERE session_id LIKE 'eval_ragas_%'")
        conn.commit()
        print("[LIMPIEZA] Registros temporales 'eval_ragas_%' eliminados de Turso.")
    except Exception as e:
        print(
            f"[ADVERTENCIA] No se pudieron limpiar registros temporales en Turso: {e}")


async def main():
    if not os.path.exists(ARCHIVO_CSV):
        print(f"Error: No se encontró el archivo {ARCHIVO_CSV}")
        return

    df = pd.read_csv(ARCHIVO_CSV)
    total = len(df)
    print(f"=== INICIANDO CAPTURA DE EVALUACIÓN ({total} PREGUNTAS) ===\n")

    registros_capturados = []

    for idx, fila in df.iterrows():
        q_id = str(fila["id"])
        tipo = str(fila["tipo"])
        categoria = str(fila["categoria"])
        pregunta = str(fila["pregunta"])
        ground_truth = str(fila["respuesta_referencia"])
        fuente = str(fila["fuente"])
        nota = "" if pd.isna(fila["nota"]) else str(fila["nota"])

        session_id = f"eval_ragas_{q_id}_{int(time.time())}"
        print(f"[{idx + 1}/{total}] ({q_id}) {pregunta}")

        # A. Ejecutar Advanced RAG (COODIBOT)
        t0 = time.time()
        res_rag = await procesar_rag(pregunta, session_id)
        latencia_rag = round(time.time() - t0, 3)

        respuesta_rag = res_rag.get("texto", "")
        contextos_rag = res_rag.get("contextos", [])
        oas_vinculados = res_rag.get("oas_vinculados", [])

        if not contextos_rag:
            contextos_rag = [
                "Sin fragmentos recuperados por encima del umbral de similitud."]

        # B. Ejecutar Línea Base Zero-Shot (sin RAG)
        t1 = time.time()
        respuesta_zeroshot = await generar_respuesta_zero_shot(pregunta)
        latencia_zeroshot = round(time.time() - t1, 3)

        registros_capturados.append({
            "id": q_id,
            "tipo": tipo,
            "categoria": categoria,
            "fuente": fuente,
            "nota": nota,
            "question": pregunta,
            "ground_truth": ground_truth,
            "answer_rag": respuesta_rag,
            "contexts_rag": contextos_rag,
            "oas_vinculados": oas_vinculados,
            "latencia_rag_seg": latencia_rag,
            "answer_zeroshot": respuesta_zeroshot,
            "contexts_zeroshot": ["Sin recuperación documental (modelo Zero-Shot)."],
            "latencia_zeroshot_seg": latencia_zeroshot
        })

        with open(ARCHIVO_SALIDA_JSON, "w", encoding="utf-8") as f:
            json.dump(registros_capturados, f, ensure_ascii=False, indent=2)

    limpiar_sesiones_evaluacion()
    print(f"\n=== CAPTURA COMPLETADA CON ÉXITO ===")
    print(f"Dataset guardado en: {ARCHIVO_SALIDA_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
