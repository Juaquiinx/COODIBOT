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

# Corregido al nombre de tu índice real en la nube
indice = pc.Index("coodibot-memoria")

# 3. Configuración de carpetas y archivos
CARPETA_PDFS = "documentos_coodi"
RUTA_EXCEL_OA = "tabla4_OA_vinculados_COODI.xlsx"

# Separador semántico de LangChain
separador_inteligente = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)


def extraer_y_picar_pdfs(carpeta):
    """Lee todos los PDFs, inyecta metadatos del Excel y convierte cada página en fragmentos"""
    fragmentos_totales = []

    # Intentar cargar la Tabla 4 de OA
    try:
        df_oa = pd.read_excel(RUTA_EXCEL_OA)
    except Exception as e:
        print(f"Advertencia: No se pudo cargar {RUTA_EXCEL_OA}. Error: {e}")
        df_oa = pd.DataFrame()

    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            print(f"Leyendo documento: {nombre_archivo}...")

            with open(ruta_completa, "rb") as archivo_pdf:
                lector = PyPDF2.PdfReader(archivo_pdf)

                for i, pagina in enumerate(lector.pages):
                    texto_extraido = pagina.extract_text()

                    if texto_extraido and len(texto_extraido.strip()) > 50:
                        texto_limpio = texto_extraido.replace(
                            "\n", " ").strip()
                        texto_limpio = " ".join(texto_limpio.split())

                        # CHUNKING INTELIGENTE
                        chunks = separador_inteligente.split_text(texto_limpio)

                        for j, chunk in enumerate(chunks):
                            # Estructura base de metadatos
                            metadatos_chunk = {
                                "fuente": nombre_archivo,
                                "tipo_documento": "Documento Oficial Mineduc",
                                "pagina": str(i+1),
                                "curso": "No identificado",
                                "asignatura": "No identificada",
                                "codigo_oa": "OA no identificado"
                            }

                            # Lógica para inyectar el OA cruzando con los nombres EXACTOS de las columnas
                            if not df_oa.empty:
                                texto_lower = chunk.lower()
                                for _, fila in df_oa.iterrows():
                                    # Rescatar valores usando los encabezados de la Tabla 4
                                    nivel_excel = str(
                                        fila.get('Nivel', '')).lower()
                                    desc_excel = str(
                                        fila.get('Descripción del Objetivo de Aprendizaje (Mineduc)', '')).lower()

                                    # Usamos los primeros 40 caracteres de la descripción para evitar fallos por espacios o saltos de línea
                                    desc_corta = desc_excel[:40] if len(
                                        desc_excel) > 40 else desc_excel

                                    # Si detectamos el nivel o la descripción en el fragmento del PDF
                                    if (nivel_excel and nivel_excel in texto_lower) or (desc_corta and desc_corta in texto_lower):
                                        metadatos_chunk["curso"] = str(
                                            fila.get('Nivel', ''))
                                        metadatos_chunk["asignatura"] = str(
                                            fila.get('Asignatura', ''))
                                        metadatos_chunk["codigo_oa"] = str(
                                            fila.get('Código Oficial', ''))
                                        break

                            fragmentos_totales.append({
                                "id": f"{nombre_archivo}-pag-{i+1}-chunk-{j+1}",
                                "texto": chunk,
                                "metadatos": metadatos_chunk
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
        time.sleep(0.02)

    # 6. Subir a Pinecone en lotes de 100
    print("Subiendo vectores a Pinecone...")
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)
        print(f"Lote {i} a {i+len(lote)} subido...")

    print("¡Memoria inyectada con éxito! La Base Curricular está en la nube con un contexto perfecto.")
