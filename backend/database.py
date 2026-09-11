# Encargada de administrar la base de datos (ahora en la nube, con Turso)

import os
import asyncio
import libsql
from dotenv import load_dotenv

load_dotenv()

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")


def _conectar():
    """Abre una conexion nueva a la base de datos remota de Turso."""
    return libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)


# ---------------------------------------------------------------------------
# Cada funcion publica es "async def" para no romper nada en main.py/rag_service.py
# (que ya hacen "await guardar_mensaje(...)", etc). Por dentro, el trabajo real
# lo hace una funcion sincrona que corre en un hilo aparte (asyncio.to_thread),
# porque el paquete libsql de Turso todavia no tiene una version async nativa.
# ---------------------------------------------------------------------------

def _iniciar_base_datos_sync():
    conn = _conectar()
    conn.execute('''
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


async def iniciar_base_datos():
    """Crea la tabla de historial en Turso si no existe."""
    await asyncio.to_thread(_iniciar_base_datos_sync)


def _guardar_mensaje_sync(session_id, rol, contenido, pinecone_ids):
    conn = _conectar()
    cursor = conn.execute(
        "INSERT INTO historial_chat (session_id, rol, contenido, calificacion, pinecone_ids) VALUES (?, ?, ?, NULL, ?)",
        (session_id, rol, contenido, pinecone_ids)
    )
    conn.commit()
    return cursor.lastrowid


async def guardar_mensaje(session_id: str, rol: str, contenido: str, pinecone_ids: str = None):
    """Guarda un mensaje en Turso y devuelve su id."""
    return await asyncio.to_thread(_guardar_mensaje_sync, session_id, rol, contenido, pinecone_ids)


def _obtener_historial_sync(session_id, limite):
    conn = _conectar()
    filas = conn.execute(
        "SELECT rol, contenido FROM historial_chat WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limite)
    ).fetchall()
    return [{"role": fila[0], "content": fila[1]} for fila in reversed(filas)]


async def obtener_historial(session_id: str, limite: int = 4):
    """Recupera los ultimos mensajes de una sesion desde Turso."""
    return await asyncio.to_thread(_obtener_historial_sync, session_id, limite)


def _actualizar_calificacion_sync(mensaje_id, calificacion):
    conn = _conectar()
    conn.execute("UPDATE historial_chat SET calificacion = ? WHERE id = ?", (calificacion, mensaje_id))
    conn.commit()


async def actualizar_calificacion(mensaje_id: int, calificacion: int):
    """Guarda el like/dislike (o -1) de una respuesta en Turso."""
    await asyncio.to_thread(_actualizar_calificacion_sync, mensaje_id, calificacion)


def _obtener_respuestas_malas_sync():
    conn = _conectar()
    filas = conn.execute(
        "SELECT id, contenido, pinecone_ids FROM historial_chat WHERE calificacion = -1"
    ).fetchall()
    return [{"mensaje_id": f[0], "contenido": f[1], "pinecone_ids": f[2]} for f in filas]


async def obtener_respuestas_malas():
    """Recupera las respuestas con calificacion negativa, para el panel admin."""
    return await asyncio.to_thread(_obtener_respuestas_malas_sync)