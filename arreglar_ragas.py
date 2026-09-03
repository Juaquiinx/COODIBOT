# Script temporal: arregla un import roto dentro de la libreria ragas instalada en venv_ragas.
# Se puede borrar despues de usarlo una vez (no se sube a GitHub).

import os

ruta = os.path.join("venv_ragas", "Lib", "site-packages", "ragas", "llms", "base.py")

with open(ruta, "r", encoding="utf-8") as f:
    contenido = f.read()

viejo = "from langchain_community.chat_models.vertexai import ChatVertexAI"
nuevo = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "except ImportError:\n"
    "    class ChatVertexAI:  # no instalado, no lo usamos (solo usamos OpenAI)\n"
    "        pass\n"
)

if viejo not in contenido:
    print("No encontre la linea a reemplazar. Puede que ya este arreglado, o el archivo cambio.")
else:
    contenido_nuevo = contenido.replace(viejo, nuevo, 1)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido_nuevo)
    print("Listo, arreglado.")