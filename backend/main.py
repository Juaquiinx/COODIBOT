# Importar herramientas
import os
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Cargar las llaves ocultas
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Inicializar los clientes
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
indice = pc.Index("coodibot-memoria")

# Cargar el diccionario de OAs para traducir códigos a texto
diccionario_oas = {}
try:
    with open("catalogo_oas.json", "r", encoding="utf-8") as f:
        catalogo = json.load(f)
        for asignatura, lista_oas in catalogo.items():
            for oa in lista_oas:
                diccionario_oas[oa["id"]] = oa["descripcion"]
except Exception as e:
    print(f"Advertencia: No se pudo cargar el catálogo JSON: {e}")

# Inicialización de la base de datos local


def iniciar_base_datos():
    """Crea la base de datos SQLite y la tabla de historial si no existen"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial_chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rol TEXT,
            contenido TEXT,
            calificacion INTEGER,
            pinecone_ids TEXT 
        )
    ''')
    conn.commit()
    conn.close()


def guardar_mensaje(session_id: str, rol: str, contenido: str, pinecone_ids: str = None):
    """Guarda un mensaje en la base de datos, incluyendo los IDs de Pinecone si es el bot"""
    conn = sqlite3.connect("memoria_coodibot.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO historial_chat (session_id, rol, contenido, calificacion, pinecone_ids) VALUES (?, ?, ?, NULL, ?)",
                   (session_id, rol, contenido, pinecone_ids))
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

# Inicializar la API
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


class CorreccionOA(BaseModel):
    mensaje_id: int
    pinecone_ids: str
    nuevo_oa: str


@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El servidor de COODIBOT está en línea, escuchando y recordando!"}

# Lógica Advanced RAG con Memoria


def procesar_rag(pregunta_texto: str, session_id: str):
    print(
        f"\n[CEREBRO] Procesando consulta: '{pregunta_texto}' (Sesión: {session_id})")

    try:
        historial_reciente = obtener_historial(session_id)
        historial_str = ""
        for msg in historial_reciente:
            rol = "Profesor" if msg["role"] == "user" else "COODIBOT"
            historial_str += f"{rol}: {msg['content']}\n"

        vec_busqueda = cliente_openai.embeddings.create(
            input=pregunta_texto,
            model="text-embedding-3-small"
        ).data[0].embedding

        res_tec = indice.query(
            vector=vec_busqueda,
            top_k=7,
            include_metadata=True,
            filter={"category": "coodi_manual"}
        )

        contexto_recuperado = ""
        fragmentos_utilizados = 0
        oa_oficial_extraido = ""
        lista_codigos_guardados = []
        ids_pinecone_utilizados = []

        for match in res_tec.matches:
            score = match.score

            if score >= 0.15:
                texto = match.metadata.get("texto", "")
                contexto_recuperado += texto + "\n\n---\n\n"
                fragmentos_utilizados += 1

                ids_pinecone_utilizados.append(match.id)

                codigo = match.metadata.get("codigo_oa", "Ninguno")
                if codigo != "Ninguno" and codigo != "OA no identificado" and oa_oficial_extraido == "":
                    lista_codigos = [c.strip() for c in codigo.split(",")]
                    lista_codigos_validos = [
                        c for c in lista_codigos if c in diccionario_oas]

                    if lista_codigos_validos:
                        lista_codigos_guardados = lista_codigos_validos
                        lista_vinetas = [
                            f"- {c}" for c in lista_codigos_validos]
                        oa_oficial_extraido = "\n           ".join(
                            lista_vinetas)

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
            id_mensaje_vacio = guardar_mensaje(
                session_id, "assistant", respuesta_sin_datos, ids_pinecone_str)
            return {"texto": respuesta_sin_datos, "mensaje_id": id_mensaje_vacio}

        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.
        
        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        3. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
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

        respuesta_llm = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=mensajes_finales,
            temperature=0.1
        )

        respuesta_final = respuesta_llm.choices[0].message.content

        for c in lista_codigos_guardados:
            desc = diccionario_oas.get(c, "")
            if desc:
                html_tooltip = f'<span class="coodi-tooltip">{c} 👁️<span class="coodi-tooltip-text"><b>{c}:</b> {desc}</span></span>'
                respuesta_final = respuesta_final.replace(c, html_tooltip)

        id_mensaje = guardar_mensaje(
            session_id, "assistant", respuesta_final, ids_pinecone_str)

        return {"texto": respuesta_final, "mensaje_id": id_mensaje}

    except Exception as e:
        print(f"ERROR EN RAG: {str(e)}")
        raise e

# Rutas de la api del chat


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

# Rutas para el panel de administrador


@app.get("/api/admin/malas")
def obtener_respuestas_malas():
    """Devuelve todas las respuestas calificadas con -1 por el usuario."""
    print("\n[ADMIN] Solicitando lista de malas respuestas...")
    try:
        conn = sqlite3.connect("memoria_coodibot.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, contenido, pinecone_ids FROM historial_chat WHERE calificacion = -1")
        filas = cursor.fetchall()
        conn.close()

        resultados = [{"mensaje_id": f[0], "contenido": f[1],
                       "pinecone_ids": f[2]} for f in filas]
        return {"estado": "éxito", "data": resultados}
    except Exception as e:
        return {"error": f"Error al obtener respuestas: {str(e)}"}


@app.put("/api/admin/corregir-oa")
def corregir_oa_en_pinecone(datos: CorreccionOA):
    """Actualiza la metadata en Pinecone directamente y limpia el error en SQLite."""
    print(
        f"\n[ADMIN] Corrigiendo OA en Pinecone para el mensaje {datos.mensaje_id}...")
    try:
        if not datos.pinecone_ids:
            return {"error": "No hay IDs de Pinecone asociados a esta respuesta."}

        lista_ids = datos.pinecone_ids.split(",")

        for pinecone_id in lista_ids:
            indice.update(id=pinecone_id, set_metadata={
                          "codigo_oa": datos.nuevo_oa})
            print(
                f"Vector {pinecone_id} actualizado en Pinecone a: {datos.nuevo_oa}")

        conn = sqlite3.connect("memoria_coodibot.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE historial_chat SET calificacion = 2 WHERE id = ?",
                       (datos.mensaje_id,))  # 2 = Corregido
        conn.commit()
        conn.close()

        return {"estado": "éxito", "mensaje": "Metadata en Pinecone actualizada correctamente. ¡El bot ha aprendido!"}
    except Exception as e:
        return {"error": f"Error al actualizar Pinecone: {str(e)}"}
