# El cerebro activo de COODIBOT, hace las conexiones correspondientes

# Importar herramientas
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models import MensajeUsuario, EvaluacionRespuesta, CorreccionOA
from database import iniciar_base_datos, actualizar_calificacion, obtener_respuestas_malas
from rag_service import procesar_rag, actualizar_metadata_pinecone, sugerir_oas_con_confianza


@asynccontextmanager
async def lifespan(app: FastAPI):
    await iniciar_base_datos()
    yield

# Configuración principal de la aplicación
app = FastAPI(title="COODIBOT API", lifespan=lifespan)

# Configuración de CORS para permitir solicitudes de cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Desde aquí parten los ENDPOINTS o Rutas de las API


@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El servidor de COODIBOT está en línea, modularizado, asíncrono y seguro!"}


# Recibe la pregunta del docente y devuelve la respuesta armada por el pipeline RAG
@app.post("/api/chat")
async def chatear_texto(mensaje: MensajeUsuario):
    print("\n[RUTA] Ingreso por TEXTO detectado")
    try:
        respuesta = await procesar_rag(mensaje.pregunta, mensaje.session_id)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": f"Hubo un problema procesando la consulta: {str(e)}"}


# El docente califica una respuesta (bien/mal) desde el chat
@app.put("/api/chat/evaluar")
async def evaluar_respuesta(evaluacion: EvaluacionRespuesta):
    print(
        f"\n[EVALUACIÓN] Recibiendo nota {evaluacion.calificacion} para el mensaje {evaluacion.mensaje_id}")
    try:
        await actualizar_calificacion(evaluacion.mensaje_id, evaluacion.calificacion)
        return {"estado": "éxito", "mensaje": "Evaluación guardada correctamente"}
    except Exception as e:
        return {"error": f"Hubo un problema al guardar la evaluación: {str(e)}"}


# Panel admin: trae las respuestas mal calificadas con Top-3 OAs sugeridos y métrica de ambigüedad
@app.get("/api/admin/malas")
async def obtener_respuestas_malas_endpoint():
    print("\n[ADMIN] Solicitando lista de malas respuestas con análisis de confianza OA...")
    try:
        resultados = await obtener_respuestas_malas()
        for item in resultados:
            texto_analisis = item.get(
                "pregunta_usuario") or item.get("contenido", "")
            analisis_oa = await sugerir_oas_con_confianza(texto_analisis, top_k=3, tau=0.05)
            item["sugerencias_oa"] = analisis_oa["sugerencias"]
            item["ambiguedad"] = analisis_oa["ambiguedad"]

        return {"estado": "éxito", "data": resultados}
    except Exception as e:
        return {"error": f"Error al obtener respuestas: {str(e)}"}


# Panel admin: el admin corrige el OA de una respuesta mala directo en Pinecone
@app.put("/api/admin/corregir-oa")
async def corregir_oa_en_pinecone_endpoint(datos: CorreccionOA):
    print(
        f"\n[ADMIN] Corrigiendo OA en Pinecone para el mensaje {datos.mensaje_id}...")
    try:
        if not datos.pinecone_ids:
            return {"error": "No hay IDs de Pinecone asociados a esta respuesta."}

        actualizar_metadata_pinecone(datos.pinecone_ids, datos.nuevo_oa)
        # marca la respuesta como corregida
        await actualizar_calificacion(datos.mensaje_id, 2)

        return {"estado": "éxito", "mensaje": "Metadata en Pinecone actualizada correctamente."}
    except Exception as e:
        return {"error": f"Error al actualizar Pinecone: {str(e)}"}
