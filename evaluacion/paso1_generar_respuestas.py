"""
PASO 1 de la evaluacion RAGAS: le hace las preguntas a COODIBOT y guarda las respuestas.
Corre en el entorno normal del proyecto (el mismo de main.py), sin instalar nada nuevo.
"""

import os
import sys
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CSV_PREGUNTAS = os.path.join(CARPETA_ACTUAL, "preguntas_ragas.csv")
CSV_SALIDA = os.path.join(CARPETA_ACTUAL, "respuestas_coodibot_crudo.csv")
LIMITE_PREGUNTAS = None  # Empezamos en modo prueba; luego lo cambiamos a None para las 50


def leer_preguntas():
    with open(CSV_PREGUNTAS, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    if LIMITE_PREGUNTAS:
        filas = filas[:LIMITE_PREGUNTAS]
        print(f"MODO PRUEBA: usando solo las primeras {LIMITE_PREGUNTAS} preguntas.")
    return filas


def main():
    from main import procesar_rag

    preguntas = leer_preguntas()
    filas_resultado = []

    for fila in preguntas:
        pregunta_id = fila["id"]
        pregunta_texto = fila["pregunta"]
        print(f"[{pregunta_id}] Preguntando: {pregunta_texto[:70]}...")
        session_id = f"ragas_eval_{pregunta_id}"

        try:
            resultado = procesar_rag(pregunta_texto, session_id)
        except Exception as e:
            print(f"  -> ERROR procesando {pregunta_id}: {e}")
            continue

        respuesta_texto = resultado.get("texto", "")
        contextos = resultado.get("contextos", [])
        if not contextos:
            print(f"  -> AVISO: {pregunta_id} no recupero ningun contexto")

        filas_resultado.append({
            "id": pregunta_id,
            "tipo": fila["tipo"],
            "user_input": pregunta_texto,
            "response": respuesta_texto,
            "retrieved_contexts": " ||| ".join(contextos),
            "reference": fila["respuesta_referencia"],
        })
        time.sleep(0.3)

    if not filas_resultado:
        print("No se obtuvo ningun resultado de COODIBOT.")
        sys.exit(1)

    with open(CSV_SALIDA, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tipo", "user_input", "response", "retrieved_contexts", "reference"])
        writer.writeheader()
        writer.writerows(filas_resultado)

    print(f"\nListo. Respuestas guardadas en: {CSV_SALIDA}")
    print("Ahora corre el PASO 2 dentro del entorno virtual venv_ragas.")


if __name__ == "__main__":
    main()