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
# INICIALIZACIÓN DE LA BASE DE DATOS LOCAL (MEMORIA)
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
    cursor.execute("INSERT INTO historial_chat (session_id, rol, contenido, calificacion) VALUES (?, ?, ?, NULL)",
                   (session_id, rol, contenido))
    ultimo_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ultimo_id


def obtener_historial(session_id: str, limite: int = 4):
    """Recupera los últimos mensajes de la conversación para dar contexto"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT rol, contenido FROM historial_chat WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                   (session_id, limite))
    filas = cursor.fetchall()
    conn.close()
    return [{"role": fila[0], "content": fila[1]} for fila in reversed(filas)]


iniciar_base_datos()
# =====================================================================

# 3. Inicializar la API
app = FastAPI(title="COODIBOT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MensajeUsuario(BaseModel):
    pregunta: str
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
    print(
        f"\n[CEREBRO] Procesando consulta: '{pregunta_texto}' (Sesión: {session_id})")

    try:
        historial_reciente = obtener_historial(session_id)
        historial_str = ""
        for msg in historial_reciente:
            rol = "Profesor" if msg["role"] == "user" else "COODIBOT"
            historial_str += f"{rol}: {msg['content']}\n"

        # =====================================================================
        # PASO 1 PRE-RETRIEVAL: Multi-Query Routing (Advanced RAG)
        # =====================================================================
        prompt_limpieza = f"""
        Actúa como un enrutador de búsqueda semántica.
        Historial reciente:
        {historial_str}
        Pregunta actual: "{pregunta_texto}"
        
        REGLA ESTRICTA: Separa la intención de búsqueda en DOS consultas divididas por un símbolo "|".
        Consulta 1: Solo conceptos técnicos de hardware o programación.
        Consulta 2: Solo nivel escolar y asignatura (pedagógico). Si no hay, escribe "General".
        
        Ejemplo de salida: sensor ultrasónico HC-SR04 | 5° básico tecnología
        """

        respuesta_limpieza = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_limpieza}],
            temperature=0.0
        )

        # Dividimos las consultas generadas por el LLM
        consultas_raw = respuesta_limpieza.choices[0].message.content.strip().split(
            "|")
        consulta_tecnica = consultas_raw[0].strip()
        consulta_curricular = consultas_raw[1].strip() if len(
            consultas_raw) > 1 else "General"

        print(f"[CEREBRO] Búsqueda Técnica: '{consulta_tecnica}'")
        print(f"[CEREBRO] Búsqueda Curricular: '{consulta_curricular}'")

        guardar_mensaje(session_id, "user", pregunta_texto)

        # =====================================================================
        # PASO 2 y 3: Búsqueda Vectorial Paralela
        # =====================================================================
        resultados_totales = []

        # Búsqueda 1: Manuales Técnicos
        if consulta_tecnica:
            vec_tec = cliente_openai.embeddings.create(
                input=consulta_tecnica, model="text-embedding-3-small").data[0].embedding
            res_tec = indice.query(
                vector=vec_tec, top_k=10, include_metadata=True)
            resultados_totales.extend(res_tec.matches)

        # Búsqueda 2: Bases Curriculares Mineduc
        if consulta_curricular and consulta_curricular.lower() != "general":
            vec_curr = cliente_openai.embeddings.create(
                input=consulta_curricular, model="text-embedding-3-small").data[0].embedding
            res_curr = indice.query(
                vector=vec_curr, top_k=10, include_metadata=True)
            resultados_totales.extend(res_curr.matches)

        # =====================================================================
        # PASO 4: Fusión de Contexto y Extracción de OA
        # =====================================================================
        contexto_recuperado = ""
        fragmentos_utilizados = 0
        oa_oficial_extraido = "OA no identificado"

        # Ordenamos todos los resultados combinados por su score de similitud
        resultados_totales = sorted(
            resultados_totales, key=lambda x: x.score, reverse=True)

        for match in resultados_totales:
            score = match.score
            texto = match.metadata.get("texto", "")

            # EXTRACCIÓN DEL METADATO OA
            codigo = match.metadata.get("codigo_oa", "OA no identificado")
            if codigo != "OA no identificado" and oa_oficial_extraido == "OA no identificado":
                asignatura = match.metadata.get("asignatura", "")
                curso = match.metadata.get("curso", "")
                oa_oficial_extraido = f"{codigo} - {asignatura} {curso}".strip(
                    "- ")

            # Umbral de similitud (puedes bajarlo a 0.15 si es necesario)
            if score >= 0.15:
                contexto_recuperado += texto + "\n\n---\n\n"
                fragmentos_utilizados += 1

        print(f"Fragmentos que superaron el umbral: {fragmentos_utilizados}")
        print(f"OA Recuperado de la base: {oa_oficial_extraido}")

        if fragmentos_utilizados == 0:
            respuesta_sin_datos = "No tengo información sobre esto en mis manuales oficiales."
            id_mensaje_vacio = guardar_mensaje(
                session_id, "assistant", respuesta_sin_datos)
            return {"texto": respuesta_sin_datos, "mensaje_id": id_mensaje_vacio}

        # =====================================================================
        # PASO 5: Generación Final
        # =====================================================================
        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.
        
        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        3. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
           - Concepto Clave: (Definición breve)
           - Pasos: (Instrucciones numeradas con verbos imperativos)
           - OA Vinculado: {oa_oficial_extraido}
           - Verificación: (Cómo comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto_recuperado}
        """

        mensajes_finales = [{"role": "system", "content": prompt_sistema}]
        mensajes_finales.extend(historial_reciente)
        mensajes_finales.append({"role": "user", "content": pregunta_texto})

        respuesta_llm = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=mensajes_finales,
            temperature=0.1
        )

        respuesta_final = respuesta_llm.choices[0].message.content
        id_mensaje = guardar_mensaje(session_id, "assistant", respuesta_final)

        return {"texto": respuesta_final, "mensaje_id": id_mensaje}

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
    print(
        f"\n[EVALUACIÓN] Recibiendo nota {evaluacion.calificacion} para el mensaje {evaluacion.mensaje_id}")
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
