import os
import time
import PyPDF2
import pandas as pd
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
indice = pc.Index("coodibot-memoria")

# 3. Configuración de carpetas y archivos
CARPETA_PDFS = "documentos_coodi"
RUTA_EXCEL_OA = "tabla4_OA_vinculados_COODI.xlsx"

separador_inteligente = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)


def procesar_pdfs_tecnicos(carpeta):
    """Convierte los manuales PDF en fragmentos técnicos puros."""
    fragmentos = []
    if not os.path.exists(carpeta):
        return fragmentos

    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            print(f"Leyendo documento técnico: {nombre_archivo}...")
            with open(ruta_completa, "rb") as archivo_pdf:
                lector = PyPDF2.PdfReader(archivo_pdf)
                for i, pagina in enumerate(lector.pages):
                    texto = pagina.extract_text()
                    if texto and len(texto.strip()) > 50:
                        texto_limpio = " ".join(
                            texto.replace("\n", " ").split())
                        for j, chunk in enumerate(separador_inteligente.split_text(texto_limpio)):
                            fragmentos.append({
                                "id": f"pdf-{nombre_archivo}-pag-{i+1}-chunk-{j+1}",
                                "texto": chunk,
                                "metadatos": {
                                    "fuente": nombre_archivo,
                                    "tipo_documento": "Documento Oficial",
                                    "pagina": str(i+1),
                                    "category": "coodi_manual",
                                    "codigo_oa": "OA no identificado"
                                }
                            })
    return fragmentos


def procesar_excel_oas(ruta_excel):
    """Transforma cada fila del Excel en un chunk independiente (El Cerebro Curricular)."""
    fragmentos = []
    try:
        df_oa = pd.read_excel(ruta_excel)
        for index, fila in df_oa.iterrows():
            # Extraemos los datos usando los nombres exactos de tus columnas
            curso = str(fila.get('Nivel', 'Sin nivel'))
            asignatura = str(fila.get('Asignatura', 'Sin asignatura'))
            codigo = str(fila.get('Código Oficial', 'OA general'))
            descripcion = str(
                fila.get('Descripción del Objetivo de Aprendizaje (Mineduc)', ''))

            # Armamos un bloque de texto rico en contexto
            texto_chunk = f"Objetivo de Aprendizaje ({codigo}) de {asignatura} para {curso}. Descripción: {descripcion}."

            fragmentos.append({
                "id": f"excel-oa-fila-{index+1}",
                "texto": texto_chunk,
                "metadatos": {
                    "fuente": ruta_excel,
                    "category": "coodi_curriculum",
                    "curso": curso,
                    "asignatura": asignatura,
                    "codigo_oa": codigo
                }
            })
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el Excel. Error: {e}")
    return fragmentos


# 4. Ejecutar la extracción unificada
print("Buscando documentos y procesando currículum...")
textos_para_procesar = procesar_pdfs_tecnicos(
    CARPETA_PDFS) + procesar_excel_oas(RUTA_EXCEL_OA)

if not textos_para_procesar:
    print("¡No se encontraron documentos válidos para inyectar!")
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

        vectores_para_subir.append({
            "id": item["id"],
            "values": respuesta.data[0].embedding,
            "metadata": {"texto": item["texto"], **item["metadatos"]}
        })
        time.sleep(0.02)

    # 6. Subir a Pinecone en lotes de 100
    print("Subiendo vectores a Pinecone...")
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)
        print(f"Lote {i} a {i+len(lote)} subido...")

    print("¡Memoria inyectada con éxito! La Base Curricular está en la nube con un contexto perfecto.")
