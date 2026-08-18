import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
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


@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El servidor de COODIBOT está en línea y escuchando!"}

# =====================================================================
# EL CEREBRO COMPARTIDO: Lógica Advanced RAG
# =====================================================================


def procesar_rag(pregunta_texto: str):
    """Esta función recibe texto, reformula la consulta, busca en Pinecone y genera la respuesta"""
    print(f"\n[CEREBRO] Procesando consulta original: '{pregunta_texto}'")

    try:
        # =====================================================================
        # NUEVO PASO PRE-RETRIEVAL: Reformulación de Consulta (Query Rewriting)
        # =====================================================================
        prompt_limpieza = f"""
        Actúa como un extractor de conceptos clave. 
        Tu objetivo es leer la pregunta de un profesor, ignorar todo el ruido conversacional (saludos, muletillas, contexto innecesario) y extraer SOLO los conceptos técnicos, asignaturas y cursos.
        Devuelve una frase corta y robótica ideal para buscar en una base de datos vectorial.
        
        Pregunta del profesor: "{pregunta_texto}"
        """

        respuesta_limpieza = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_limpieza}],
            temperature=0.1
        )
        consulta_optimizada = respuesta_limpieza.choices[0].message.content.strip(
        )
        print(
            f"[CEREBRO] Consulta optimizada para Pinecone: '{consulta_optimizada}'")
        # =====================================================================

        # PASO A: Vectorizar la consulta OPTIMIZADA (No la original)
        respuesta_embedding = cliente_openai.embeddings.create(
            input=consulta_optimizada,
            model="text-embedding-3-small"
        )
        vector_pregunta = respuesta_embedding.data[0].embedding

        # PASO B: Recuperación en Pinecone
        resultados_busqueda = indice.query(
            vector=vector_pregunta,
            top_k=10,
            include_metadata=True
        )

        # PASO C: Armar el Contexto (Con el umbral en 0.20)
        contexto_recuperado = ""
        fragmentos_utilizados = 0

        print("--- RESULTADOS PINECONE ---")
        for match in resultados_busqueda.matches:
            score = match.score
            texto = match.metadata.get("texto", "")

            if score >= 0.20:
                contexto_recuperado += texto + "\n\n---\n\n"
                fragmentos_utilizados += 1

        print(f"Fragmentos que superaron el umbral: {fragmentos_utilizados}")

        if fragmentos_utilizados == 0:
            return "No tengo información sobre esto en mis manuales."

        # ====== AGREGA ESTOS 3 PRINTS AQUÍ ======
        print("\n\n=============== RAYOS X: EL CONTEXTO ================")
        print(contexto_recuperado)
        print("=====================================================\n\n")
        # ========================================

        # PASO D: Generación con OpenAI (Microaprendizaje)
        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.
        
        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Si la respuesta no está en el contexto, di "No tengo información sobre esto en mis manuales".
        3. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        4. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
           - Concepto Clave: (Definición breve)
           - Pasos: (Instrucciones numeradas con verbos imperativos)
           - OA Vinculado: (Código y descripción del OA)
           - Verificación: (Cómo comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto_recuperado}
        """

        # Le pasamos la pregunta_texto ORIGINAL para que responda con empatía a lo que el profesor realmente dijo
        respuesta_llm = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_texto}
            ],
            temperature=0.1
        )

        return respuesta_llm.choices[0].message.content

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
        respuesta = procesar_rag(mensaje.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"Hubo un problema procesando la consulta: {str(e)}"}

# =====================================================================
# RUTA 2: NUEVA ENTRADA POR AUDIO (WHISPER)
# =====================================================================


@app.post("/api/chat/audio")
async def chatear_audio(file: UploadFile = File(...)):
    print(f"\n[RUTA] Ingreso por AUDIO detectado. Archivo: {file.filename}")

    # 1. Guardar el audio temporalmente en el servidor
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Enviar a Whisper para transcribir (Speech-to-Text)
        print("Enviando audio a Whisper de OpenAI...")
        with open(temp_file_path, "rb") as audio_file:
            transcripcion = cliente_openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es"  # Forzamos español para mayor precisión
            )

        texto_del_profesor = transcripcion.text
        print(f"Whisper escuchó: '{texto_del_profesor}'")

        # 3. Borrar el archivo temporal para no llenar tu disco duro
        os.remove(temp_file_path)

        # 4. Pasar el texto transcrito al "Cerebro" (RAG)
        respuesta_rag = procesar_rag(texto_del_profesor)

        # 5. Devolver la transcripción y la respuesta al frontend
        return {
            "transcripcion": texto_del_profesor,
            "respuesta": respuesta_rag
        }

    except Exception as e:
        # Limpiar archivo en caso de error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {"error": f"Hubo un problema con el audio: {str(e)}"}
