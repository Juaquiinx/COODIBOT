# Encargada de administrar la base de datos

# Importación de la base de datos asincrónica
import aiosqlite

DB_NAME = "memoria_coodibot.db"

# Inicialización de la base datos


async def iniciar_base_datos():
    """Crea la base de datos SQLite y la tabla de historial de forma asíncrona"""
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS historial_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                rol TEXT,
                contenido TEXT,
                calificacion INTEGER,
                pinecone_ids TEXT 
            )
        ''')
        await db.commit()


async def guardar_mensaje(session_id: str, rol: str, contenido: str, pinecone_ids: str = None):
    """Guarda un mensaje en la base de datos sin bloquear otras peticiones"""
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute(
            "INSERT INTO historial_chat (session_id, rol, contenido, calificacion, pinecone_ids) VALUES (?, ?, ?, NULL, ?)",
            (session_id, rol, contenido, pinecone_ids)
        )
        ultimo_id = cursor.lastrowid
        await db.commit()
        return ultimo_id


async def obtener_historial(session_id: str, limite: int = 4):
    """Recupera los últimos mensajes sin bloquear lectura concurrente"""
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute(
            "SELECT rol, contenido FROM historial_chat WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limite)
        )
        filas = await cursor.fetchall()
        return [{"role": fila[0], "content": fila[1]} for fila in reversed(filas)]


async def actualizar_calificacion(mensaje_id: int, calificacion: int):
    """Actualiza la nota sin interrumpir a otros usuarios"""
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE historial_chat SET calificacion = ? WHERE id = ?", (calificacion, mensaje_id))
        await db.commit()


async def obtener_respuestas_malas():
    """Recupera respuestas negativas para el panel admin"""
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute("SELECT id, contenido, pinecone_ids FROM historial_chat WHERE calificacion = -1")
        filas = await cursor.fetchall()
        return [{"mensaje_id": f[0], "contenido": f[1], "pinecone_ids": f[2]} for f in filas]
