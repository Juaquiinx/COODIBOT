import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 1. Cargar las llaves ocultas desde el archivo .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("¡Ojo! No se encontró la llave de Pinecone. Revisa tu archivo .env")

# 2. Inicializar el cliente de Pinecone
print("Conectando a Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# 3. Definir el nombre de nuestro índice y sus características
nombre_indice = "coodibot-memoria"

# Revisamos si el índice ya existe para no crearlo dos veces
if nombre_indice not in pc.list_indexes().names():
    print(f"Creando el índice '{nombre_indice}'... (Esto puede tardar unos segundos)")
    pc.create_index(
        name=nombre_indice,
        dimension=1536, # Este es el tamaño exacto de los vectores del modelo de OpenAI (text-embedding-3-small)
        metric="cosine", # Métrica de similitud de coseno, la que justificaste en tu informe
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1" # Región gratuita por defecto de Pinecone
        )
    )
    print("¡Índice creado con éxito en la nube!")
else:
    print(f"El índice '{nombre_indice}' ya existe. ¡Listo para usarse!")