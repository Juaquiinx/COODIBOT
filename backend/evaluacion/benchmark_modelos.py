import asyncio
import json
import time
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelos de OpenAI a comparar
MODELOS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

# Selección de 10 casos representativos del dataset (técnicos, curriculares y trampa)
IDS_MUESTRA = ["T01", "T09", "T16", "T17",
               "T20", "C02", "C09", "C15", "C18", "C20"]

with open("dataset_capturado_50.json", "r", encoding="utf-8") as f:
    dataset = [row for row in json.load(f) if row["id"] in IDS_MUESTRA]


async def evaluar_modelo(modelo: str):
    resultados = []
    for item in dataset:
        contexto = "\n\n---\n\n".join(item["contexts_rag"])
        oas_txt = "\n           ".join(
            [f"- {c}" for c in item["oas_vinculados"]]) or "Ninguno (Consulta puramente técnica)"

        prompt_sistema = f"""Eres COODIBOT, un asistente experto en robótica educativa.
REGLAS ESTRICTAS:
1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
2. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
3. Las OAs solo existen para los cursos: primero, segundo, tercero, cuarto, quinto y sexto básico.
4. Si se te pregunta por algun curso fuera del rango de entre primero y sexto basico, sugiere OAs de cursos mas bajos dejando claro que no es el curso solicitado.
5. Si el contexto NO menciona el componente, indica que no tienes información o que podría estar planificado pero no implementado aún.
6. OBLIGATORIO: Sigue EXACTAMENTE esta estructura de 4 partes:
   - Concepto Clave: (Definición breve)
   - Pasos: (Instrucciones numeradas con verbos imperativos)
   - OA Vinculado:
   {oas_txt}
   - Verificación: (Cómo comprobar que funcionó)

CONTEXTO RECUPERADO DE LOS MANUALES:
{contexto}"""

        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": item["question"]}
            ],
            temperature=0.1
        )
        latencia = time.perf_counter() - t0
        texto = resp.choices[0].message.content

        # Validación automática de formato (si cumple las 4 etiquetas de colores del frontend)
        cumple_formato = all(k in texto for k in [
                             "Concepto Clave:", "Pasos:", "OA Vinculado:", "Verificación:"])
        palabras = len(texto.split())

        resultados.append({
            "modelo": modelo,
            "id": item["id"],
            "latencia_seg": round(latencia, 3),
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
            "palabras": palabras,
            "cumple_formato": cumple_formato,
            "respuesta": texto
        })
    return resultados


async def main():
    todo = []
    for m in MODELOS:
        print(f"Evaluando {m}...")
        todo.extend(await evaluar_modelo(m))

    df = pd.DataFrame(todo)
    df.to_csv("comparativa_modelos_detalle.csv", index=False)

    resumen = df.groupby("modelo").agg(
        latencia_media_s=("latencia_seg", "mean"),
        tokens_in_prom=("tokens_in", "mean"),
        tokens_out_prom=("tokens_out", "mean"),
        palabras_prom=("palabras", "mean"),
        cumplimiento_formato_pct=(
            "cumple_formato", lambda x: round(x.mean() * 100, 1))
    ).round(3)
    print("\n=== RESUMEN COMPARATIVO DE MODELOS ===")
    print(resumen.to_string())

asyncio.run(main())
