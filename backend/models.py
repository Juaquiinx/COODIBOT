# Creación de clases para mantener un orden
from pydantic import BaseModel


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
