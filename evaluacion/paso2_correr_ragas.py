"""
PASO 2 de la evaluacion RAGAS: calcula las 4 metricas sobre las respuestas del Paso 1.
IMPORTANTE: este script se corre DENTRO del entorno virtual "venv_ragas", no en el normal.
"""

import os
import csv
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CSV_CRUDO = os.path.join(CARPETA_ACTUAL, "respuestas_coodibot_crudo.csv")

UMBRALES_TESIS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
    "context_precision": 0.80,
}


def leer_respuestas_crudas():
    filas = []
    with open(CSV_CRUDO, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            contextos = fila["retrieved_contexts"].split(" ||| ") if fila["retrieved_contexts"] else []
            filas.append({
                "id": fila["id"], "tipo": fila["tipo"], "user_input": fila["user_input"],
                "response": fila["response"], "retrieved_contexts": contextos, "reference": fila["reference"],
            })
    return filas


def correr_ragas(filas_resultado: list):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    muestras = [
        SingleTurnSample(
            user_input=f["user_input"], response=f["response"],
            retrieved_contexts=f["retrieved_contexts"] or [""], reference=f["reference"],
        )
        for f in filas_resultado
    ]
    dataset_eval = EvaluationDataset(samples=muestras)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metricas = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    print("\nCorriendo RAGAS sobre", len(muestras), "preguntas...")
    return evaluate(dataset=dataset_eval, metrics=metricas)


def guardar_y_resumir(resultado_ragas, filas_resultado):
    df_resultado = resultado_ragas.to_pandas()
    df_resultado["id"] = [f["id"] for f in filas_resultado]
    df_resultado["tipo"] = [f["tipo"] for f in filas_resultado]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ruta_salida = os.path.join(CARPETA_ACTUAL, f"resultados_ragas_{timestamp}.csv")
    df_resultado.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    print(f"\nResultados guardados en: {ruta_salida}")

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
            print(f"  {nombre_umbral}: columna '{col}' no encontrada")
            continue
        promedio = df_resultado[col].mean()
        umbral = UMBRALES_TESIS[nombre_umbral]
        estado = "CUMPLE" if promedio >= umbral else "NO CUMPLE"
        print(f"  {nombre_umbral:20s} promedio={promedio:.3f}   umbral_tesis={umbral:.2f}   -> {estado}")

    print("=" * 60)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY en el archivo .env.")
        return
    if not os.path.exists(CSV_CRUDO):
        print(f"No encontre {CSV_CRUDO}. Corre primero paso1_generar_respuestas.py.")
        return
    filas_resultado = leer_respuestas_crudas()
    resultado_ragas = correr_ragas(filas_resultado)
    guardar_y_resumir(resultado_ragas, filas_resultado)


if __name__ == "__main__":
    main()