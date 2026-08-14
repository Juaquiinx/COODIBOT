import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. Cargar las llaves ocultas
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Inicializar los clientes (El cerebro de OpenAI y la memoria de Pinecone)
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
indice = pc.Index("coodibot-memoria")

# 3. Preparar nuestros primeros "Chunks" (Fragmentos) basados en tu tesis
textos_prueba = [
    {
        "id": "oa-cien-5b-10",
        "texto": "OA 10: Analizar un circuito eléctrico simple (pila, cables, ampolleta, interruptor) para comprender su funcionamiento.",
        "metadatos": {
            "nivel_educativo": "Educación Básica",
            "curso": "5° Básico",
            "asignatura": "Ciencias Naturales",
            "tipo_documento": "Bases Curriculares Mineduc"
        }
    },
    {
        "id": "manual-coodi-ultra",
        "texto": "SENSOR ULTRASÓNICO HC-SR04: Mide distancias. Conecta el pin VCC al puerto de 5V del COODI. Conecta GND a tierra. Conecta TRIG al pin digital 9 y ECHO al pin 10.",
        "metadatos": {
            "nivel_educativo": "Educación Básica",
            "curso": "Transversal",
            "asignatura": "Tecnología",
            "tipo_documento": "Manual Técnico COODI"
        }
    }
]

# 4. Convertir texto a matemáticas (Embeddings) y subir a la nube
print("Vectorizando textos con OpenAI e inyectando a Pinecone...")
vectores_para_subir = []

for item in textos_prueba:
    # A) OpenAI convierte el texto crudo en un vector de 1536 dimensiones
    respuesta = cliente_openai.embeddings.create(
        input=item["texto"],
        model="text-embedding-3-small"
    )
    vector = respuesta.data[0].embedding

    # B) Preparamos el paquete exacto que Pinecone exige: (ID, Vector, Metadatos)
    # Nota importante: Guardamos el texto original adentro de los metadatos para poder leerlo después
    vectores_para_subir.append({
        "id": item["id"],
        "values": vector,
        "metadata": {"texto": item["texto"], **item["metadatos"]}
    })

# C) Upsert (Insertar o Actualizar) en la base de datos
indice.upsert(vectors=vectores_para_subir)
print("¡Memoria inyectada con éxito! COODIBOT ya tiene conocimientos almacenados.")