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

# MEJORA CLAVE: Función para dividir textos largos en chunks (pedazos) más pequeños con solapamiento.


def dividir_texto(texto, max_caracteres=800, solapamiento=150):
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + max_caracteres
        fragmentos.append(texto[inicio:fin])
        # El solapamiento evita cortar ideas por la mitad
        inicio += (max_caracteres - solapamiento)
    return fragmentos


def extraer_y_picar_pdfs(carpeta):
    """Lee todos los PDFs y convierte cada página en un fragmento seguro y digerible"""
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
                        texto_limpio = texto_extraido.replace(
                            "\n", " ").strip()

                        # CHUNKING: Cortamos la página en pedazos manejables para la IA
                        chunks = dividir_texto(texto_limpio)

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
        f"Se generaron {len(textos_para_procesar)} fragmentos de conocimiento. Vectorizando...")

    # OPCIONAL PERO RECOMENDADO: Limpiar la base de datos antes de subir lo nuevo
    # print("Limpiando vectores antiguos en Pinecone...")
    # indice.delete(delete_all=True)
    # time.sleep(3)

    vectores_para_subir = []

    # 5. Convertir a Embeddings y empaquetar
    for i, item in enumerate(textos_para_procesar):
        # Mostramos progreso cada 100 chunks para no llenar la terminal
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

        # Tiempo prudente para no reventar la API de OpenAI
        time.sleep(0.02)

    # 6. Subir a Pinecone en lotes de 100
    print("Subiendo vectores a Pinecone...")
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)
        print(f"Lote {i} a {i+len(lote)} subido...")

    print("¡Memoria inyectada con éxito! La Base Curricular ya está en la nube de Pinecone.")
