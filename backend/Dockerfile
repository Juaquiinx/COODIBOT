# Usar una imagen oficial y ligera de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar la lista de dependencias primero (para optimizar la caché de Docker)
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de tu proyecto al contenedor (excepto lo del .dockerignore)
COPY . .

# Exponer el puerto por donde escuchará FastAPI
EXPOSE 8000

# El comando para encender el servidor cuando el contenedor inicie
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]