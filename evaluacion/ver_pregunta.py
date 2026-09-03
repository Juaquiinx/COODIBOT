"""
Muestra el detalle completo (pregunta, respuesta de COODIBOT, contexto recuperado,
respuesta de referencia) para IDs de pregunta especificos, para revisarlos a mano.

Como correrlo (entorno normal, sin venv_ragas activado):
  py evaluacion\ver_pregunta.py
"""
import os
import csv

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CSV_CRUDO = os.path.join(CARPETA_ACTUAL, "respuestas_coodibot_crudo.csv")

IDS_A_REVISAR = ["C10", "C20", "C07", "C09", "C06"]  # cambia esta lista para revisar otras preguntas

with open(CSV_CRUDO, newline="", encoding="utf-8-sig") as f:
    filas = {fila["id"]: fila for fila in csv.DictReader(f)}

for id_pregunta in IDS_A_REVISAR:
    fila = filas.get(id_pregunta)
    if not fila:
        print(f"No encontre la pregunta {id_pregunta}\n")
        continue

    print("=" * 70)
    print(f"ID: {id_pregunta}  (tipo: {fila['tipo']})")
    print("=" * 70)
    print(f"\nPREGUNTA:\n{fila['user_input']}")
    print(f"\nRESPUESTA DE COODIBOT:\n{fila['response']}")
    print(f"\nRESPUESTA DE REFERENCIA (lo esperado):\n{fila['reference']}")
    contextos = fila['retrieved_contexts'].split(" ||| ") if fila['retrieved_contexts'] else []
    print(f"\nCONTEXTO RECUPERADO ({len(contextos)} fragmentos):")
    if not contextos or contextos == [""]:
        print("  (NO SE RECUPERO NINGUN FRAGMENTO)")
    else:
        for i, c in enumerate(contextos, 1):
            recorte = c[:500] + ("..." if len(c) > 500 else "")
            print(f"\n  --- Fragmento {i} ---\n  {recorte}")
    print("\n")