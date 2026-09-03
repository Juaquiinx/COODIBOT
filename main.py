import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. Cargar las llaves ocultas
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Inicializar los clientes (OpenAI y Pinecone)
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
indice = pc.Index("coodibot-memoria")

# =====================================================================
# NUEVO: INICIALIZACIÓN DE LA BASE DE DATOS LOCAL (MEMORIA)
# =====================================================================


def iniciar_base_datos():
    """Crea la base de datos SQLite y la tabla de historial si no existen"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS historial_chat')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial_chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rol TEXT,
            contenido TEXT,
            calificacion INTEGER
        )
    ''')
    conn.commit()
    conn.close()


def guardar_mensaje(session_id: str, rol: str, contenido: str):
    """Guarda un mensaje en la base de datos y devuelve su ID"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    # Insertamos el mensaje dejando la calificación en NULL por defecto
    cursor.execute("INSERT INTO historial_chat (session_id, rol, contenido, calificacion) VALUES (?, ?, ?, NULL)",
                   (session_id, rol, contenido))
    ultimo_id = cursor.lastrowid # Capturamos el número asignado
    conn.commit()
    conn.close()

    return ultimo_id


def obtener_historial(session_id: str, limite: int = 4):
    """Recupera los últimos mensajes de la conversación para dar contexto"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    # Traemos los últimos X mensajes ordenados por ID
    cursor.execute("SELECT rol, contenido FROM historial_chat WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                   (session_id, limite))
    filas = cursor.fetchall()
    conn.close()
    # Invertimos la lista para que el mensaje más antiguo de la muestra quede primero
    return [{"role": fila[0], "content": fila[1]} for fila in reversed(filas)]


# Ejecutamos la inicialización al arrancar el servidor
iniciar_base_datos()
# =====================================================================

# 3. Inicializar la API
app = FastAPI(title="COODIBOT API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Definir la estructura del mensaje de texto


class MensajeUsuario(BaseModel):
    pregunta: str
    # Agregamos session_id por defecto para no romper tu frontend actual
    session_id: str = "sesion_docente_default"

class EvaluacionRespuesta(BaseModel):
    mensaje_id: int
    calificacion: int

@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El servidor de COODIBOT está en línea, escuchando y recordando!"}

# =====================================================================
# EL CEREBRO COMPARTIDO: Lógica Advanced RAG con Memoria
# =====================================================================


def procesar_rag(pregunta_texto: str, session_id: str):
    """Esta función recibe texto, revisa el historial, reformula la consulta, busca en Pinecone y genera la respuesta"""
    print(
        f"\n[CEREBRO] Procesando consulta: '{pregunta_texto}' (Sesión: {session_id})")

    try:
        # Recuperar el historial de chat de esta sesión
        historial_reciente = obtener_historial(session_id)

        # Convertir historial a texto para que el LLM lo lea
        historial_str = ""
        for msg in historial_reciente:
            rol = "Profesor" if msg["role"] == "user" else "COODIBOT"
            historial_str += f"{rol}: {msg['content']}\n"

        # =====================================================================
        # PASO 1 PRE-RETRIEVAL: Reformulación de Consulta "Context-Aware"
        # =====================================================================
        prompt_limpieza = f"""
        Actúa como un extractor de conceptos clave de búsqueda.
        Aquí tienes el historial reciente de la conversación:
        ---
        {historial_str}
        ---
        Pregunta actual del profesor: "{pregunta_texto}"

        REGLAS:
        1. Si la pregunta actual hace referencia a algo del historial (ej: "explica el punto 2", "dame más detalles"), usa el historial para entender a qué se refiere y crea UNA SOLA frase de búsqueda completa.
        2. Ignora saludos, gracias, y ruido.
        3. Devuelve SOLO los conceptos técnicos, asignaturas y cursos. Sin comillas ni texto extra.
        """

        respuesta_limpieza = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",  # Modelo rápido y barato para lógica interna
            messages=[{"role": "user", "content": prompt_limpieza}],
            temperature=0.0
        )
        consulta_optimizada = respuesta_limpieza.choices[0].message.content.strip(
        )
        print(
            f"[CEREBRO] Consulta optimizada para Pinecone: '{consulta_optimizada}'")
        # =====================================================================

        # Guardamos la pregunta del usuario en la base de datos (después de la limpieza, guardamos la original para ser fieles)
        guardar_mensaje(session_id, "user", pregunta_texto)

        # PASO 2: Vectorizar la consulta OPTIMIZADA (No la original)
        respuesta_embedding = cliente_openai.embeddings.create(
            input=consulta_optimizada,
            model="text-embedding-3-small"
        )
        vector_pregunta = respuesta_embedding.data[0].embedding

        # PASO 3: Recuperación en Pinecone
        resultados_busqueda = indice.query(
            vector=vector_pregunta,
            top_k=10,
            include_metadata=True
        )

        # PASO 4: Armar el Contexto Curricular
        contexto_recuperado = ""
        contextos_lista = []  # NUEVO: lista de fragmentos individuales (para evaluación RAGAS)
        fragmentos_utilizados = 0

        for match in resultados_busqueda.matches:
            score = match.score
            texto = match.metadata.get("texto", "")

            # Umbral de similitud
            if score >= 0.20:
                contexto_recuperado += texto + "\n\n---\n\n"
                contextos_lista.append(texto)  # NUEVO
                fragmentos_utilizados += 1

        print(f"Fragmentos que superaron el umbral: {fragmentos_utilizados}")

        if fragmentos_utilizados == 0:
            respuesta_sin_datos = "No tengo información sobre esto en mis manuales oficiales."
            # MODIFICADO: Guardamos y capturamos el ID incluso si no hay datos
            id_mensaje_vacio = guardar_mensaje(session_id, "assistant", respuesta_sin_datos)
            # NUEVO: incluimos "contextos" (vacío) para mantener la misma forma de respuesta siempre
            return {"texto": respuesta_sin_datos, "mensaje_id": id_mensaje_vacio, "contextos": []}

        # PASO 5: Generación con OpenAI (Microaprendizaje + Contexto Conversacional)
        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.

        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del CONTEXTO RECUPERADO de abajo. No agregues datos, cifras ni afirmaciones que no estén en el contexto.
        2. Si el contexto contiene información relacionada con la pregunta (aunque sea parcial o no la responda de forma perfecta), úsala para responder de la mejor forma posible. No es necesario que el contexto conteste la pregunta textualmente para que puedas responder.
        3. Indica que no tienes información SOLO si el contexto no menciona en absoluto el tema de la pregunta. En ese caso responde brevemente: "No cuento con información sobre esto en mis manuales."
        4. OA Vinculado: cita un código de Objetivo de Aprendizaje SOLO si aparece explícitamente escrito en el contexto recuperado. Si no aparece ningún código, escribe "OA no identificado en el material recuperado" — NUNCA inventes ni deduzcas un código de OA.
        5. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje), salvo que estés indicando que no tienes información (ahí una frase basta).
        6. Cuando sí respondas con contenido, sigue EXACTAMENTE esta estructura de 4 partes:
            - Concepto Clave: (Definición breve)
            - Pasos: (Instrucciones numeradas con verbos imperativos)
            - OA Vinculado: (Código y descripción del OA, o "no identificado" según regla 4)
            - Verificación: (Cómo comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto_recuperado}
        """

        # Preparamos los mensajes para el LLM final, inyectando el historial para fluidez
        mensajes_finales = [{"role": "system", "content": prompt_sistema}]

        # Le pasamos el historial previo para que tenga memoria de la conversación
        mensajes_finales.extend(historial_reciente)

        # Finalmente, agregamos la pregunta actual
        mensajes_finales.append({"role": "user", "content": pregunta_texto})

        respuesta_llm = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=mensajes_finales,
            temperature=0.1
        )

        respuesta_final = respuesta_llm.choices[0].message.content

        # MODIFICADO: Guardamos la respuesta de COODIBOT capturando su ID
        id_mensaje = guardar_mensaje(session_id, "assistant", respuesta_final)

        # Retornamos el diccionario esperado por las rutas
        # NUEVO: agregamos "contextos" (lista de fragmentos usados), útil para depuración y para evaluación RAGAS
        return {"texto": respuesta_final, "mensaje_id": id_mensaje, "contextos": contextos_lista}

    except Exception as e:
        print(f"ERROR EN RAG: {str(e)}")
        raise e

# =====================================================================
# RUTA 1: ENTRADA POR TEXTO TRADICIONAL
# =====================================================================


@app.post("/api/chat")
def chatear_texto(mensaje: MensajeUsuario):
    print("\n[RUTA] Ingreso por TEXTO detectado")
    try:
        respuesta = procesar_rag(mensaje.pregunta, mensaje.session_id)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"Hubo un problema procesando la consulta: {str(e)}"}


@app.put("/api/chat/evaluar")
def evaluar_respuesta(evaluacion: EvaluacionRespuesta):
    print(f"\n[EVALUACIÓN] Recibiendo nota {evaluacion.calificacion} para el mensaje {evaluacion.mensaje_id}")
    try:
        conn = sqlite3.connect("memoria_coodibot.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE historial_chat SET calificacion = ? WHERE id = ?",
                       (evaluacion.calificacion, evaluacion.mensaje_id))
        conn.commit()
        conn.close()
        return {"estado": "éxito", "mensaje": "Evaluación guardada correctamente"}
    except Exception as e:
        return {"error": f"Hubo un problema al guardar la evaluación: {str(e)}"}
