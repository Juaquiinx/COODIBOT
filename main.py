import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # IMPORTACIÓN NUEVA
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

# ESTO ES NUEVO: Configuración de CORS para permitir que la web se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # El "*" permite que cualquier web se conecte. Ideal para fase de desarrollo.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Definir la estructura del mensaje
class MensajeUsuario(BaseModel):
    pregunta: str

@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El servidor de COODIBOT está en línea y conectado a la IA!"}

# 5. EL NÚCLEO: La ruta del chat con Advanced RAG
@app.post("/api/chat")
def chatear(mensaje: MensajeUsuario):
    pregunta_texto = mensaje.pregunta

    try:
        # PASO A: Vectorizar la pregunta del profesor
        respuesta_embedding = cliente_openai.embeddings.create(
            input=pregunta_texto,
            model="text-embedding-3-small"
        )
        vector_pregunta = respuesta_embedding.data[0].embedding

        # PASO B: Recuperación (Retrieval) en Pinecone
        # Aumentamos a 10 fragmentos para tener más contexto del Mineduc
        resultados_busqueda = indice.query(
            vector=vector_pregunta,
            top_k=10, 
            include_metadata=True
        )

        # PASO C: Armar el Contexto
        # Juntamos los textos de los manuales que encontró Pinecone
        contexto_recuperado = ""
        for match in resultados_busqueda["matches"]:
            contexto_recuperado += match["metadata"]["texto"] + "\n\n"

        # PASO D: Generación (Generation) con OpenAI
        # Aquí aplicamos tu regla de Microaprendizaje y la plantilla de salida
        prompt_sistema = f"""
        Eres COODIBOT, un asistente experto en robótica educativa.
        Tu objetivo es ayudar a docentes de educación básica.
        
        REGLAS ESTRICTAS:
        1. Responde SIEMPRE basándote ÚNICAMENTE en la información del contexto proporcionado.
        2. Si la respuesta no está en el contexto, di "No tengo información sobre eso en mis manuales".
        3. Mantén tu respuesta por debajo de las 100 palabras (Microaprendizaje).
        4. OBLIGATORIO: Tu respuesta debe seguir EXACTAMENTE esta estructura de 4 partes:
           - Concepto Clave: (Definición breve en 1 o 2 oraciones)
           - Pasos: (Instrucciones numeradas con verbos imperativos)
           - OA Vinculado: (Menciona el código y descripción del OA si está en el contexto)
           - Verificación: (Cómo el profesor puede comprobar que funcionó)

        CONTEXTO RECUPERADO DE LOS MANUALES:
        {contexto_recuperado}
        """

        respuesta_llm = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_texto}
            ],
            temperature=0.1 # Temperatura baja para que sea preciso y no alucine
        )

        respuesta_final = respuesta_llm.choices[0].message.content

        return {"respuesta": respuesta_final}

    except Exception as e:
        return {"error": f"Hubo un problema procesando la consulta: {str(e)}"}