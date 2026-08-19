import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeException

# 1. Cargar las llaves de tu archivo .env
load_dotenv()

# 2. Conectarse a Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# OJO: Verifica que "coodibot-memoria" sea el nombre exacto de tu índice
nombre_indice = "coodibot-memoria"
indice = pc.Index(nombre_indice)

print(f"Conectando al índice '{nombre_indice}' y buscando fragmentos...\n")

# 3. Crear un vector "falso" lleno de ceros.
# El modelo text-embedding-3-small usa 1536 dimensiones.
vector_falso = [0.0] * 1536

# 4. Consultar a Pinecone
try:
    resultados = indice.query(
        vector=vector_falso,
        top_k=3,  # Traeremos solo 3 fragmentos de ejemplo
        include_metadata=True  # ¡Esto es lo más importante! Trae el texto legible
    )

    print("=== MUESTRA DE FRAGMENTOS EN PINECONE ===\n")
    for i, match in enumerate(resultados['matches']):
        print(f"--- Fragmento {i+1} ---")
        print(f"ID: {match['id']}")

        print("Metadatos (El contenido real que lee el bot):")
        # Imprimir los metadatos de forma ordenada
        if 'metadata' in match:
            for clave, valor in match['metadata'].items():
                print(f"  > {clave}: {valor}")
        else:
            print(
                "  > [ALERTA] Este vector no tiene metadatos. El bot no puede leer nada aquí.")
        print("\n" + "="*40 + "\n")

except PineconeException as e:
    print(f"Ocurrió un error al consultar Pinecone: {e}")
