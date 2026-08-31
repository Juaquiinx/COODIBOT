"""
Evaluación automatizada de COODIBOT con RAGAS.

Que hace:
  1. Lee las 50 preguntas de referencia (evaluacion/preguntas_ragas.csv).
  2. Le hace cada pregunta a COODIBOT (llamando directamente a la funcion
     procesar_rag de main.py, sin necesidad de tener el servidor corriendo aparte).
  3. Junto con la respuesta de COODIBOT, guarda los fragmentos que recupero de Pinecone
     (el "contexto"), necesarios para que RAGAS pueda calcular sus 4 metricas.
  4. Corre RAGAS (Faithfulness, Answer/Response Relevancy, Context Recall, Context Precision)
     y guarda un CSV con el resultado por pregunta + un resumen comparado contra
     los umbrales minimos que se definieron en la Tabla 5 de la tesis.

Como correrlo:
  cd COODIBOT
  pip install -r requirements.txt
  python evaluacion/evaluar_ragas.py

Para probar rapido con pocas preguntas antes de gastar cuota de la API en las 50,
hay que cambiar LIMITE_PREGUNTAS mas abajo (por ejemplo a 5) y volver a dejarlo en None
cuando queramos correr el set completo.
"""

import os
import sys
import time
import csv
from datetime import datetime

# Permite importar main.py, que esta en la carpeta raiz del proyecto (un nivel arriba de evaluacion/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ============================================================
# CONFIGURACION
# ============================================================
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CSV_PREGUNTAS = os.path.join(CARPETA_ACTUAL, "preguntas_ragas.csv")
LIMITE_PREGUNTAS = None  # Poner un numero (ej. 5) para pruebas rapidas; None = las 50 completas

UMBRALES_TESIS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
    "context_precision": 0.80,
}

# ============================================================
# PASO 1: Correr las preguntas contra COODIBOT
# ============================================================


def ejecutar_preguntas_en_coodibot(df_preguntas: pd.DataFrame):
    """Le pasa cada pregunta a procesar_rag() de main.py y guarda respuesta + contextos."""
    from main import procesar_rag  # import tardio: usa las llaves de .env ya cargadas

    filas_resultado = []

    for _, fila in df_preguntas.iterrows():
        pregunta_id = fila["id"]
        pregunta_texto = fila["pregunta"]
        referencia = fila["respuesta_referencia"]

        print(f"[{pregunta_id}] Preguntando: {pregunta_texto[:70]}...")

        session_id = f"ragas_eval_{pregunta_id}"  # sesion unica por pregunta: sin memoria cruzada entre preguntas

        try:
            resultado = procesar_rag(pregunta_texto, session_id)
        except Exception as e:
            print(f"  -> ERROR procesando {pregunta_id}: {e}")
            continue

        respuesta_texto = resultado.get("texto", "")
        contextos = resultado.get("contextos", [])

        if not contextos:
            print(f"  -> AVISO: {pregunta_id} no recupero ningun contexto (revisar si el corpus esta cargado)")

        filas_resultado.append({
            "id": pregunta_id,
            "tipo": fila["tipo"],
            "user_input": pregunta_texto,
            "response": respuesta_texto,
            "retrieved_contexts": contextos,
            "reference": referencia,
        })

        time.sleep(0.3)  # pequena pausa para no saturar la API

    return filas_resultado


# ============================================================
# PASO 2: Correr RAGAS sobre los resultados
# ============================================================


def correr_ragas(filas_resultado: list):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    muestras = [
        SingleTurnSample(
            user_input=f["user_input"],
            response=f["response"],
            retrieved_contexts=f["retrieved_contexts"] or [""],  # ragas no acepta lista vacia
            reference=f["reference"],
        )
        for f in filas_resultado
    ]
    dataset_eval = EvaluationDataset(samples=muestras)

    # Modelo "juez" que usa RAGAS para calificar (independiente del modelo generador de COODIBOT)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metricas = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    print("\nCorriendo RAGAS sobre", len(muestras), "preguntas... (puede tardar varios minutos)")
    resultado = evaluate(dataset=dataset_eval, metrics=metricas)
    return resultado


# ============================================================
# PASO 3: Guardar y mostrar resumen
# ============================================================


def guardar_y_resumir(resultado_ragas, filas_resultado):
    df_resultado = resultado_ragas.to_pandas()

    # Agregamos el id y tipo original para poder cruzar los resultados con preguntas_ragas.csv
    df_resultado["id"] = [f["id"] for f in filas_resultado]
    df_resultado["tipo"] = [f["tipo"] for f in filas_resultado]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ruta_salida = os.path.join(CARPETA_ACTUAL, f"resultados_ragas_{timestamp}.csv")
    df_resultado.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    print(f"\nResultados detallados guardados en: {ruta_salida}")

    print("\n" + "=" * 60)
    print("RESUMEN vs. UMBRALES DE LA TABLA 5 DE LA TESIS")
    print("=" * 60)

    columnas_metricas = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy" if "answer_relevancy" in df_resultado.columns else "response_relevancy",
        "context_recall": "context_recall",
        "context_precision": "context_precision" if "context_precision" in df_resultado.columns else "llm_context_precision_with_reference",
    }

    for nombre_umbral, col in columnas_metricas.items():
        if col not in df_resultado.columns:
            print(f"  {nombre_umbral}: columna '{col}' no encontrada en el resultado, revisar nombres de metrica en esta version de ragas")
            continue
        promedio = df_resultado[col].mean()
        umbral = UMBRALES_TESIS[nombre_umbral]
        estado = "CUMPLE" if promedio >= umbral else "NO CUMPLE"
        print(f"  {nombre_umbral:20s} promedio={promedio:.3f}   umbral_tesis={umbral:.2f}   -> {estado}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================


def main():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("Faltan OPENAI_API_KEY o PINECONE_API_KEY en el archivo .env. Revisa el README del proyecto.")
        sys.exit(1)

    df_preguntas = pd.read_csv(CSV_PREGUNTAS)
    if LIMITE_PREGUNTAS:
        df_preguntas = df_preguntas.head(LIMITE_PREGUNTAS)
        print(f"MODO PRUEBA: usando solo las primeras {LIMITE_PREGUNTAS} preguntas.")

    filas_resultado = ejecutar_preguntas_en_coodibot(df_preguntas)

    if not filas_resultado:
        print("No se obtuvo ningun resultado de COODIBOT, no se puede evaluar con RAGAS.")
        sys.exit(1)

    # Guardamos tambien las respuestas crudas, por si RAGAS falla y hay que revisar a mano
    ruta_crudo = os.path.join(CARPETA_ACTUAL, "respuestas_coodibot_crudo.csv")
    with open(ruta_crudo, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tipo", "user_input", "response", "retrieved_contexts", "reference"])
        writer.writeheader()
        for fila in filas_resultado:
            fila_csv = dict(fila)
            fila_csv["retrieved_contexts"] = " ||| ".join(fila["retrieved_contexts"])
            writer.writerow(fila_csv)
    print(f"Respuestas crudas de COODIBOT guardadas en: {ruta_crudo}")

    resultado_ragas = correr_ragas(filas_resultado)
    guardar_y_resumir(resultado_ragas, filas_resultado)


if __name__ == "__main__":
    main()
