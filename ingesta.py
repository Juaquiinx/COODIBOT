import os
import time
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. Cargar las llaves ocultas
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Inicializar los clientes
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
indice = pc.Index("coodibot-memoria")

# 3. Configuración de la carpeta de PDFs
CARPETA_PDFS = "documentos_coodi"

def extraer_y_picar_pdfs(carpeta):
    """Lee todos los PDFs y convierte cada página en un fragmento seguro"""
    fragmentos_totales = []
    
    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            print(f"Leyendo documento: {nombre_archivo}...")
            
            with open(ruta_completa, "rb") as archivo_pdf:
                lector = PyPDF2.PdfReader(archivo_pdf)
                
                # CHUNKING POR PÁGINA: Extraemos y limpiamos página por página
                for i, pagina in enumerate(lector.pages):
                    texto_extraido = pagina.extract_text()
                    
                    # Evitar páginas vacías o con muy poco texto
                    if texto_extraido and len(texto_extraido.strip()) > 50:
                        # Limpiamos saltos de línea raros para que el texto sea fluido
                        texto_limpio = texto_extraido.replace("\n", " ").strip()
                        
                        fragmentos_totales.append({
                            "id": f"{nombre_archivo}-pag-{i+1}",
                            "texto": texto_limpio,
                            "metadatos": {
                                "fuente": nombre_archivo,
                                "tipo_documento": "Documento Oficial Mineduc",
                                "pagina": str(i+1)
                            }
                        })
    return fragmentos_totales

# 4. Ejecutar la extracción
print("Buscando documentos en la carpeta local...")
textos_para_procesar = extraer_y_picar_pdfs(CARPETA_PDFS)

if not textos_para_procesar:
    print("¡No se encontraron PDFs válidos en 'documentos_coodi'!")
else:
    print(f"Se generaron {len(textos_para_procesar)} fragmentos de conocimiento. Vectorizando...")

    vectores_para_subir = []

    # 5. Convertir a Embeddings y empaquetar
    for i, item in enumerate(textos_para_procesar):
        print(f"Procesando página {i+1} de {len(textos_para_procesar)}...")
        
        respuesta = cliente_openai.embeddings.create(
            input=item["texto"],
            model="text-embedding-3-small"
        )
        vector = respuesta.data[0].embedding

        vectores_para_subir.append({
            "id": item["id"],
            "values": vector,
            "metadata": {"texto": item["texto"], **item["metadatos"]}
        })
        
        # IMPORTANTE: Esperamos medio segundo para no saturar el límite de velocidad de OpenAI
        time.sleep(0.5)

    # 6. Subir a Pinecone (en bloques de 100 para mayor seguridad)
    print("Subiendo vectores a Pinecone...")
    # Dividimos la lista en lotes pequeños (batches) para subirlos sin problemas
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)

    print("¡Memoria inyectada con éxito! El PDF ya está en la nube de Pinecone.")