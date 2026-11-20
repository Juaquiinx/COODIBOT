import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig

# Compatibilidad de importación según versión instalada de ragas (v0.2+ o v0.1)
try:
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    USAR_CLASES_V2 = True
except ImportError:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    USAR_CLASES_V2 = False

# Rutas explícitas para backend/evaluacion/
RUTA_EVALUACION = os.path.dirname(os.path.abspath(__file__))
RUTA_BACKEND = os.path.abspath(os.path.join(RUTA_EVALUACION, ".."))
RUTA_RAIZ = os.path.abspath(os.path.join(RUTA_BACKEND, ".."))

os.chdir(RUTA_EVALUACION)

# Busca .env primero en backend/.env y luego en la raíz COODIBOT/.env
for ruta_env in [os.path.join(RUTA_BACKEND, ".env"), os.path.join(RUTA_RAIZ, ".env")]:
    if os.path.exists(ruta_env):
        load_dotenv(ruta_env)
        break

ARCHIVO_ENTRADA_JSON = os.path.join(
    RUTA_EVALUACION, "dataset_capturado_50.json")
CSV_RESULTADOS_RAG = os.path.join(
    RUTA_EVALUACION, "resultados_ragas_coodibot.csv")
CSV_RESULTADOS_BASELINE = os.path.join(
    RUTA_EVALUACION, "resultados_ragas_zeroshot.csv")
CSV_RESUMEN_COMPARATIVO = os.path.join(
    RUTA_EVALUACION, "resumen_comparativo_tesis.csv")


def construir_metricas():
    if USAR_CLASES_V2:
        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def normalizar_columnas_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas de salida de RAGAS entre v0.1 y v0.2."""
    renombres = {
        "answer_relevancy": "Answer Relevancy",
        "response_relevancy": "Answer Relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "llm_context_precision_with_reference": "Context Precision",
        "context_recall": "Context Recall",
    }
    return df.rename(columns=renombres)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No se encontró OPENAI_API_KEY en backend/.env ni en la raíz .env.")
        return

    if not os.path.exists(ARCHIVO_ENTRADA_JSON):
        print(
            f"Error: no se encontró {ARCHIVO_ENTRADA_JSON} (se genera con generar_dataset_evaluacion.py)")
        return

    with open(ARCHIVO_ENTRADA_JSON, "r", encoding="utf-8") as f:
        datos = json.load(f)

    print(
        f"Cargadas {len(datos)} interacciones desde {ARCHIVO_ENTRADA_JSON}\n")

    # LLM juez y modelo de embeddings
    llm_juez = ChatOpenAI(model="gpt-4o-mini",
                          temperature=0.0, api_key=api_key)
    embeddings_juez = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=api_key)
    config_ejecucion = RunConfig(max_workers=4, timeout=120)

    # ---------------------------------------------------------
    # 1. EVALUACIÓN DE COODIBOT (ADVANCED RAG)
    # ---------------------------------------------------------
    print(">>> Evaluando Sistema Propuesto: COODIBOT (Advanced RAG)...")
    ds_rag = Dataset.from_dict({
        "question": [d["question"] for d in datos],
        "answer": [d["answer_rag"] for d in datos],
        "contexts": [d["contexts_rag"] for d in datos],
        "ground_truth": [d["ground_truth"] for d in datos],
    })

    res_rag = evaluate(
        dataset=ds_rag,
        metrics=construir_metricas(),
        llm=llm_juez,
        embeddings=embeddings_juez,
        run_config=config_ejecucion,
    )
    df_rag = normalizar_columnas_metricas(res_rag.to_pandas())
    df_rag["id"] = [d["id"] for d in datos]
    df_rag["tipo"] = [d["tipo"] for d in datos]
    df_rag["categoria"] = [d["categoria"] for d in datos]
    df_rag["latencia_seg"] = [d["latencia_rag_seg"] for d in datos]
    df_rag.to_csv(CSV_RESULTADOS_RAG, index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 2. EVALUACIÓN DE LÍNEA BASE (ZERO-SHOT SIN RAG)
    # ---------------------------------------------------------
    print("\n>>> Evaluando Línea Base: LLM Zero-Shot (Sin RAG)...")
    ds_zero = Dataset.from_dict({
        "question": [d["question"] for d in datos],
        "answer": [d["answer_zeroshot"] for d in datos],
        "contexts": [d["contexts_zeroshot"] for d in datos],
        "ground_truth": [d["ground_truth"] for d in datos],
    })

    res_zero = evaluate(
        dataset=ds_zero,
        metrics=construir_metricas(),
        llm=llm_juez,
        embeddings=embeddings_juez,
        run_config=config_ejecucion,
    )
    df_zero = normalizar_columnas_metricas(res_zero.to_pandas())
    df_zero["id"] = [d["id"] for d in datos]
    df_zero["tipo"] = [d["tipo"] for d in datos]
    df_zero["categoria"] = [d["categoria"] for d in datos]
    df_zero["latencia_seg"] = [d["latencia_zeroshot_seg"] for d in datos]
    df_zero.to_csv(CSV_RESULTADOS_BASELINE, index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 3. TABLA RESUMEN COMPARATIVA PARA LA TESIS
    # ---------------------------------------------------------
    metricas_cols = ["Faithfulness", "Answer Relevancy",
                     "Context Precision", "Context Recall"]
    filas_resumen = []

    for m in metricas_cols:
        if m in df_rag.columns and m in df_zero.columns:
            prom_rag = df_rag[m].mean()
            std_rag = df_rag[m].std()
            prom_zero = df_zero[m].mean()
            std_zero = df_zero[m].std()
            mejora_abs = prom_rag - prom_zero
            mejora_pct = ((prom_rag - prom_zero) / prom_zero *
                          100) if prom_zero > 0 else 0.0

            filas_resumen.append({
                "Metrica": m,
                "Zero_Shot_Promedio": round(prom_zero, 4),
                "Zero_Shot_DE": round(std_zero, 4),
                "COODIBOT_RAG_Promedio": round(prom_rag, 4),
                "COODIBOT_RAG_DE": round(std_rag, 4),
                "Diferencia_Absoluta": round(mejora_abs, 4),
                "Mejora_Porcentual": f"{mejora_pct:+.1f}%"
            })

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(CSV_RESUMEN_COMPARATIVO,
                      index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("RESULTADOS COMPARATIVOS RAGAS (N = 50 PREGUNTAS)")
    print("=" * 75)
    print(df_resumen.to_string(index=False))

    print("\n--- Desglose COODIBOT RAG por Tipo de Consulta ---")
    cols_presentes = [c for c in metricas_cols if c in df_rag.columns]
    print(df_rag.groupby("tipo")[cols_presentes].mean().round(4))

    print(f"\nArchivos exportados en {RUTA_EVALUACION}:")
    print(f"  1. {os.path.basename(CSV_RESULTADOS_RAG)}")
    print(f"  2. {os.path.basename(CSV_RESULTADOS_BASELINE)}")
    print(f"  3. {os.path.basename(CSV_RESUMEN_COMPARATIVO)}")


if __name__ == "__main__":
    main()
