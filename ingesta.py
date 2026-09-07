import os
import time
import json
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
indice = pc.Index("coodibot-memoria")

# 3. Configuración de carpetas y archivos
CARPETA_PDFS = "documentos_coodi"
RUTA_CATALOGO = "catalogo_oas.json"
ARCHIVO_REGISTRO = "documentos_procesados.txt"

# Separador inteligente para los chunks
separador_inteligente = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

# 4. Funciones de Memoria (Para no gastar tokens repitiendo PDFs)


def cargar_documentos_procesados():
    """Carga la lista de PDFs que ya fueron subidos a Pinecone."""
    if not os.path.exists(ARCHIVO_REGISTRO):
        return set()
    with open(ARCHIVO_REGISTRO, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())


def registrar_documento_procesado(nombre_archivo):
    """Guarda el nombre del PDF para no volver a procesarlo en el futuro."""
    with open(ARCHIVO_REGISTRO, "a", encoding="utf-8") as f:
        f.write(f"{nombre_archivo}\n")

# 5. Cargar el Catálogo de OAs


def cargar_catalogo():
    try:
        with open(RUTA_CATALOGO, "r", encoding="utf-8") as f:
            catalogo_oas = json.load(f)
        return json.dumps(catalogo_oas, ensure_ascii=False)
    except Exception as e:
        print(f"Error al cargar el catálogo de OAs: {e}")
        return "{}"


CATALOGO_STR = cargar_catalogo()

# 6. Función Etiquetadora con IA


# 6. Función Etiquetadora con IA (Con manejo de Límites de Velocidad)
def clasificar_con_llm(chunk_texto):
    """
    Usa el LLM para leer el fragmento de COODI y asignarle los mejores OAs.
    Incluye un sistema de reintentos en caso de saturar la API de OpenAI.
    """
    prompt = f"""
    Eres un experto pedagógico de Chile. Analiza este fragmento del manual de robótica educativa COODI.
    Tu tarea es vincularlo con Objetivos de Aprendizaje (OA) del currículum chileno.
    
    Prioriza SIEMPRE asignar al menos un OA (o varios, separados por comas) si ves alguna mínima relación 
    lógica con Tecnología, Ciencias o Matemática. 
    Responde ÚNICAMENTE con los ID de los OA, sin explicaciones adicionales (Ejemplo: TEC-3B-OA1, MAT-5B-OA14).
    Si realmente es un texto puramente legal o introductorio sin NINGUNA relación, responde "Ninguno".

    Fragmento COODI:
    "{chunk_texto}"

    Catálogo de OAs disponibles:
    {CATALOGO_STR}
    """

    max_reintentos = 5
    tiempo_espera = 5  # Segundos a esperar si nos bloquean

    for intento in range(max_reintentos):
        try:
            respuesta = cliente_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            # Pausa preventiva de 1.5 segundos entre cada consulta exitosa para no saturar
            time.sleep(1.5)

            return respuesta.choices[0].message.content.strip()

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(
                    f"      [!] Límite de tokens alcanzado de OpenAI. Pausando {tiempo_espera} segundos... (Intento {intento+1}/{max_reintentos})")
                time.sleep(tiempo_espera)
                tiempo_espera += 5  # Aumenta el tiempo de espera gradualmente si insiste el bloqueo
            else:
                print(f"Error desconocido en LLM al clasificar: {e}")
                return "OA no identificado"

    return "OA no identificado (Límite excedido)"

# 7. Procesamiento de PDFs


def procesar_pdfs_tecnicos_y_etiquetar(carpeta):
    """Lee PDFs, hace chunks, pregunta a la IA el OA y prepara el empaquetado."""
    fragmentos = []
    documentos_procesados = cargar_documentos_procesados()

    if not os.path.exists(carpeta):
        print(f"La carpeta {carpeta} no existe.")
        return fragmentos

    for nombre_archivo in os.listdir(carpeta):
        if not nombre_archivo.endswith(".pdf"):
            continue

        if nombre_archivo in documentos_procesados:
            print(
                f"Saltando '{nombre_archivo}': Ya fue procesado anteriormente (ahorro de tokens).")
            continue

        ruta_completa = os.path.join(carpeta, nombre_archivo)
        print(
            f"Iniciando procesamiento y etiquetado con IA para: {nombre_archivo}...")

        with open(ruta_completa, "rb") as archivo_pdf:
            lector = PyPDF2.PdfReader(archivo_pdf)
            for i, pagina in enumerate(lector.pages):
                texto = pagina.extract_text()
                if texto and len(texto.strip()) > 50:
                    texto_limpio = " ".join(texto.replace("\n", " ").split())

                    for j, chunk in enumerate(separador_inteligente.split_text(texto_limpio)):

                        # ¡AQUÍ OCURRE LA MAGIA DEL ETIQUETADO AUTOMÁTICO!
                        print(
                            f"   -> Analizando con IA chunk {j+1} de la página {i+1}...")
                        oa_asignado = clasificar_con_llm(chunk)

                        fragmentos.append({
                            "id": f"pdf-{nombre_archivo}-pag-{i+1}-chunk-{j+1}",
                            "texto": chunk,
                            "metadatos": {
                                "fuente": nombre_archivo,
                                "tipo_documento": "Documento Oficial",
                                "pagina": str(i+1),
                                "category": "coodi_manual",
                                "codigo_oa": oa_asignado  # El metadato ahora es dinámico e inteligente
                            }
                        })

        # Anotar el documento como procesado solo si terminó con éxito
        registrar_documento_procesado(nombre_archivo)
        print(f"Documento '{nombre_archivo}' completado y registrado.")

    return fragmentos


# 8. Ejecución Principal
print("Iniciando Ingesta Inteligente de COODI...")
textos_para_procesar = procesar_pdfs_tecnicos_y_etiquetar(CARPETA_PDFS)

if not textos_para_procesar:
    print("¡No hay documentos nuevos para vectorizar e inyectar!")
else:
    print(
        f"Se generaron y etiquetaron {len(textos_para_procesar)} fragmentos. Vectorizando...")
    vectores_para_subir = []

    # Convertir a Embeddings y empaquetar
    for i, item in enumerate(textos_para_procesar):
        if i % 100 == 0 or i == len(textos_para_procesar) - 1:
            print(
                f"Vectorizando fragmento {i+1} de {len(textos_para_procesar)}...")

        respuesta = cliente_openai.embeddings.create(
            input=item["texto"],
            model="text-embedding-3-small"
        )

        vectores_para_subir.append({
            "id": item["id"],
            "values": respuesta.data[0].embedding,
            "metadata": {"texto": item["texto"], **item["metadatos"]}
        })
        time.sleep(0.02)  # Pequeña pausa para no saturar la API

    # Subir a Pinecone en lotes de 100
    print("Subiendo vectores a Pinecone...")
    tamano_lote = 100
    for i in range(0, len(vectores_para_subir), tamano_lote):
        lote = vectores_para_subir[i:i + tamano_lote]
        indice.upsert(vectors=lote)
        print(f"Lote {i} a {i+len(lote)} subido...")

    print("¡Memoria inyectada con éxito! Cada bloque del manual ahora tiene sus OAs pre-vinculados.")
