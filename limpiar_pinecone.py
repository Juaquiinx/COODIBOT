import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Inicializar cliente y seleccionar el índice
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indice = pc.Index("coodibot-memoria")

print("Eliminando todos los vectores antiguos de Pinecone...")

# Borra todo el contenido del namespace por defecto
indice.delete(delete_all=True)

print("Esperando 3 segundos a que Pinecone confirme la limpieza...")
time.sleep(3)

print("¡Listo! Tu índice 'coodibot-memoria' ha quedado 100% limpio.")
