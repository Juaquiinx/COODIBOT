# -*- coding: utf-8 -*-
"""
Analiza los resultados de RAGAS agrupando por tipo de pregunta (tecnica vs curricular),
y muestra las preguntas con peor desempeno para poder revisarlas a mano despues.

Como correrlo (entorno normal, sin venv_ragas activado):
  py evaluacion\analizar_resultados.py
"""
import os
import glob
import csv
from collections import defaultdict

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))

candidatos = glob.glob(os.path.join(CARPETA_ACTUAL, "resultados_ragas_*.csv"))
if not candidatos:
    print("No encontre ningun archivo resultados_ragas_*.csv en la carpeta evaluacion.")
    raise SystemExit(1)

ruta = max(candidatos, key=os.path.getmtime)
print(f"Analizando: {ruta}\n")

with open(ruta, newline="", encoding="utf-8-sig") as f:
    filas = list(csv.DictReader(f))

columnas_posibles = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    "context_recall": ["context_recall"],
    "context_precision": ["context_precision", "llm_context_precision_with_reference"],
}
columnas = {}
for nombre, opciones in columnas_posibles.items():
    for op in opciones:
        if filas and op in filas[0]:
            columnas[nombre] = op
            break

por_tipo = defaultdict(lambda: defaultdict(list))
for fila in filas:
    tipo = fila.get("tipo", "sin_tipo")
    for nombre, col in columnas.items():
        try:
            por_tipo[tipo][nombre].append(float(fila[col]))
        except (ValueError, KeyError):
            pass

print("=" * 70)
print("PROMEDIO POR TIPO DE PREGUNTA")
print("=" * 70)
for tipo, metricas in por_tipo.items():
    cantidad = len(next(iter(metricas.values()), []))
    print(f"\nTipo: {tipo} ({cantidad} preguntas)")
    for nombre, valores in metricas.items():
        if valores:
            promedio = sum(valores) / len(valores)
            print(f"  {nombre:20s} promedio={promedio:.3f}")

metrica_ordenar = columnas.get("faithfulness") or next(iter(columnas.values()), None)
if metrica_ordenar:
    def valor_metrica(fila):
        try:
            return float(fila[metrica_ordenar])
        except (ValueError, KeyError):
            return 1.0

    peores = sorted(filas, key=valor_metrica)[:5]

    print("\n" + "=" * 70)
    print(f"LAS 5 PREGUNTAS CON PEOR '{metrica_ordenar}'")
    print("=" * 70)
    for fila in peores:
        print(f"\nID: {fila.get('id')}  ({fila.get('tipo')})")
        print(f"  {metrica_ordenar} = {fila.get(metrica_ordenar)}")