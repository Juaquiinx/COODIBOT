# Encargado de la lógica RAG

# Importar las herramientas
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone
from database import obtener_historial, guardar_mensaje


# Carga de variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Inicialización de clientes para OpenAI y Pinecone
cliente_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
indice = pc.Index("coodibot-memoria")


# Carga de OAs
diccionario_oas = {}
try:
    with open("catalogo_oas.json", "r", encoding="utf-8") as f:
        catalogo = json.load(f)
        for asignatura, lista_oas in catalogo.items():
            for oa in lista_oas:
                diccionario_oas[oa["id"]] = oa["descripcion"]
except Exception as e:
    print(f"Advertencia: No se pudo cargar el catálogo JSON: {e}")


# Función principal de procesamiento RAG
async def procesar_rag(pregunta_texto: str, session_id: str):
    print(
        f"\n[CEREBRO] Procesando consulta: '{pregunta_texto}' (Sesión: {session_id})")

    try:
        historial_reciente = await obtener_historial(session_id)
        historial_str = ""
        for msg in historial_reciente:
            rol = "Profesor" if msg["role"] == "user" else "COODIBOT"
            historial_str += f"{rol}: {msg['content']}\n"

        vec_response = await cliente_openai.embeddings.create(
            input=pregunta_texto,
            model="text-embedding-3-small"
        )
        vec_busqueda = vec_response.data[0].embedding

        res_tec = indice.query(
            vector=vec_busqueda,
            top_k=7,
            include_metadata=True,
            filter={"category": "coodi_manual"}
        )

        contexto_recuperado = ""
        fragmentos_utilizados = 0
        oa_oficial_extraido = ""
        ids_pinecone_utilizados = []

        curso_detectado = None
        pregunta_lower = pregunta_texto.lower()
        if "6" in pregunta_lower or "sexto" in pregunta_lower:
            curso_detectado = "6B"
        elif "5" in pregunta_lower or "quinto" in pregunta_lower:
            curso_detectado = "5B"
        elif "4" in pregunta_lower or "cuarto" in pregunta_lower:
            curso_detectado = "4B"
        elif "3" in pregunta_lower or "tercero" in pregunta_lower:
            curso_detectado = "3B"
        elif "2" in pregunta_lower or "segundo" in pregunta_lower:
            curso_detectado = "2B"
        elif "1" in pregunta_lower or "primero" in pregunta_lower:
            curso_detectado = "1B"

        candidatos_pinecone = []

        for match in res_tec.matches:
            score = match.score

            if score >= 0.15:
                texto = match.metadata.get("texto", "")
                contexto_recuperado += texto + "\n\n---\n\n"
                fragmentos_utilizados += 1

                ids_pinecone_utilizados.append(match.id)

                codigo = match.metadata.get("codigo_oa", "Ninguno")
                if codigo not in ["Ninguno", "OA no identificado"]:
                    lista_codigos = [c.strip() for c in codigo.split(",")]
                    for c in lista_codigos:
                        if c in diccionario_oas and c not in candidatos_pinecone:
                            candidatos_pinecone.append(c)

        lista_codigos_guardados = []

        if curso_detectado:
            oas_del_curso = [
                c for c in candidatos_pinecone if curso_detectado in c]

            if oas_del_curso:
                lista_codigos_guardados = oas_del_curso
            else:
                oas_catalogo = [c for c in diccionario_oas.keys(
                ) if curso_detectado in c and c.startswith("TEC")]
                lista_codigos_guardados = oas_catalogo[:2]
        else:
            lista_codigos_guardados = candidatos_pinecone[:3]

        if lista_codigos_guardados:
            lista_vinetas = [f"- {c}" for c in lista_codigos_guardados]
            oa_oficial_extraido = "\n           ".join(lista_vinetas)

        for c in lista_codigos_guardados:
            desc = diccionario_oas.get(c, "")
            if desc:
                contexto_recuperado += f"\n[INFO PEDAGÓGICA OCULTA] El objetivo {c} trata sobre: {desc}\n"

        print(f"Fragmentos que superaron el umbral: {fragmentos_utilizados}")

        if not oa_oficial_extraido:
            oa_oficial_extraido = "Ninguno (Consulta puramente técnica)"

        ids_pinecone_str = ",".join(
            ids_pinecone_utilizados) if ids_pinecone_utilizados else None

        if fragmentos_utilizados == 0:
            respuesta_sin_datos = "No tengo información sobre esto en mis manuales oficiales."
            id_mensaje_vacio = await guardar_mensaje(
                session_id, "assistant", respuesta_sin_datos, ids_pinecone_str)
            return {"texto": respuesta_sin_datos, "mensaje_id": id_mensaje_vacio, "oas_vinculados": []}

        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.
        
        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        3. No respondas con objetivos de aprendizaje que no existan (fuera del rango de primero a sexto básico).
        4. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
           - Concepto Clave: (Definición breve)
           - Pasos: (Instrucciones numeradas con verbos imperativos)
           - OA Vinculado: 
           {oa_oficial_extraido}
           - Verificación: (Cómo comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto_recuperado}
        """

        mensajes_finales = [{"role": "system", "content": prompt_sistema}]
        mensajes_finales.extend(historial_reciente)
        mensajes_finales.append({"role": "user", "content": pregunta_texto})

        respuesta_llm = await cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=mensajes_finales,
            temperature=0.1
        )

        respuesta_final = respuesta_llm.choices[0].message.content

        oas_estructurados = []
        for c in lista_codigos_guardados:
            desc = diccionario_oas.get(c, "")
            if desc:
                oas_estructurados.append({"codigo": c, "descripcion": desc})

        id_mensaje = await guardar_mensaje(session_id, "assistant", respuesta_final, ids_pinecone_str)

        return {
            "texto": respuesta_final,
            "mensaje_id": id_mensaje,
            "oas_vinculados": oas_estructurados
        }

    except Exception as e:
        print(f"ERROR EN RAG: {str(e)}")
        raise e

# Gestión de metadata vectorial


def actualizar_metadata_pinecone(pinecone_ids: str, nuevo_oa: str):
    if pinecone_ids:
        lista_ids = pinecone_ids.split(",")
        for pinecone_id in lista_ids:
            indice.update(id=pinecone_id, set_metadata={"codigo_oa": nuevo_oa})
            print(
                f"Vector {pinecone_id} actualizado en Pinecone a: {nuevo_oa}")
