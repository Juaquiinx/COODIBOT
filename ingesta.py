import os
import time
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Cargar las llaves ocultas
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Inicializar los clientes
cliente_openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ¡OJO AQUÍ! Corregido al nombre de tu índice real en la nube
indice = pc.Index("coodibot-memoria")

# 3. Configuración de la carpeta de PDFs
CARPETA_PDFS = "documentos_coodi"

# MEJORA CLAVE: Reemplazamos la función manual por el separador semántico de LangChain
separador_inteligente = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    # Prioriza cortar en párrafos y puntos seguidos
    separators=["\n\n", "\n", ".", " ", ""]
)


def extraer_y_picar_pdfs(carpeta):
    """Lee todos los PDFs y convierte cada página en un fragmento con sentido semántico"""
    fragmentos_totales = []

    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            print(f"Leyendo documento: {nombre_archivo}...")

            with open(ruta_completa, "rb") as archivo_pdf:
                lector = PyPDF2.PdfReader(archivo_pdf)

                # Extraemos y limpiamos página por página
                for i, pagina in enumerate(lector.pages):
                    texto_extraido = pagina.extract_text()

                    if texto_extraido and len(texto_extraido.strip()) > 50:
                        # Limpiamos los saltos de línea raros del PDF
                        texto_limpio = texto_extraido.replace(
                            "\n", " ").strip()
                        # Quitamos espacios dobles
                        texto_limpio = " ".join(texto_limpio.split())

                        # CHUNKING INTELIGENTE: Corta manteniendo las ideas unidas
                        chunks = separador_inteligente.split_text(texto_limpio)

                        for j, chunk in enumerate(chunks):
                            fragmentos_totales.append({
                                "id": f"{nombre_archivo}-pag-{i+1}-chunk-{j+1}",
                                "texto": chunk,
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
    print(
        f"Se generaron {len(textos_para_procesar)} fragmentos con sentido. Vectorizando...")

    vectores_para_subir = []

    # 5. Convertir a Embeddings y empaquetar
    for i, item in enumerate(textos_para_procesar):
        if i % 100 == 0 or i == len(textos_para_procesar) - 1:
            print(
                f"Procesando fragmento {i+1} de {len(textos_para_procesar)}...")

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

        time.sleep(0.02)  # Respetamos el límite de API de OpenAI

    # 6. Subir a Pinecone en lotes de 100
    print("Subiendo vectores a Pinecone...")
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)
        print(f"Lote {i} a {i+len(lote)} subido...")

    print("¡Memoria inyectada con éxito! La Base Curricular está en la nube con un contexto perfecto.")
