# 🤖 COODIBOT

### Asistente educativo basado en IA para la enseñanza de robótica educativa en Chile

<p align="center">
  <strong>Advanced RAG · FastAPI · Angular · Pinecone · OpenAI · SQLite</strong>
</p>

<p align="center">
  Asistente inteligente orientado a docentes de educación básica, diseñado para apoyar la enseñanza de robótica educativa mediante recuperación semántica, generación de respuestas y curación humana de contenido.
</p>

---

## 📌 Descripción

**COODIBOT** es un asistente educativo basado en Inteligencia Artificial diseñado específicamente para **apoyar a docentes de educación básica en la enseñanza de robótica educativa en Chile**.

El proyecto utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** para combinar la recuperación de información relevante desde una base vectorial con modelos de lenguaje, permitiendo generar respuestas contextualizadas a partir de los **Objetivos de Aprendizaje (OA)** y sus respectivos contenidos.

La arquitectura integra:

* 🔎 **Búsqueda semántica vectorial** mediante Pinecone.
* 🧠 **Modelos de lenguaje de OpenAI** para generación de respuestas.
* 💾 **Memoria conversacional local** mediante SQLite.
* ⚡ **Backend REST** desarrollado con FastAPI.
* 🖥️ **Frontend modular** desarrollado con Angular.
* 📚 **Microaprendizaje** para entregar información de manera progresiva.
* 🔍 **Divulgación progresiva** de información asociada a los Objetivos de Aprendizaje.
* 👨‍💼 **Panel de administración** para la revisión y curación humana de respuestas.
* 🔄 **Actualización de información en la base vectorial de Pinecone** desde el panel de administración.

El sistema incorpora además un enfoque **Human-in-the-loop**, permitiendo que las respuestas evaluadas negativamente sean revisadas y que sus metadatos puedan ser modificados antes de actualizar la información almacenada en la base vectorial.

---

## 🏗️ Arquitectura

COODIBOT está organizado como un **monorepo**, separando la aplicación en dos componentes principales:

```text
COODIBOT/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
└── README.md
```

### Flujo general

```text
                    ┌──────────────────────┐
                    │      Docente         │
                    │     / Usuario        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Angular Frontend   │
                    │       :4200          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    │       :8000          │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │     Pinecone     │      │      SQLite      │
        │ Base vectorial   │      │ Memoria          │
        │                  │      │ conversacional   │
        └────────┬─────────┘      └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │      OpenAI      │
        │ Modelos de IA    │
        └──────────────────┘
```

El flujo RAG permite recuperar información relevante desde Pinecone antes de utilizar el modelo de lenguaje para generar la respuesta.

---

# ✨ Características principales

## 🤖 Asistente educativo

COODIBOT permite a los docentes interactuar con un asistente basado en IA orientado específicamente a la **robótica educativa**.

La interfaz incorpora un sistema de **microaprendizaje**, permitiendo presentar la información de manera progresiva en lugar de mostrar todo el contenido inmediatamente.

---

## 🔎 Arquitectura RAG

El sistema utiliza **Retrieval-Augmented Generation (RAG)** para combinar:

1. Consulta del usuario.
2. Búsqueda semántica.
3. Recuperación de información relevante.
4. Generación de una respuesta mediante un modelo de lenguaje.

La información relevante se almacena como representaciones vectoriales en **Pinecone**, permitiendo realizar búsquedas semánticas sobre los contenidos disponibles.

---

## 🧠 Modelos de lenguaje

COODIBOT utiliza **OpenAI** como componente de generación de lenguaje.

El modelo recibe el contexto recuperado desde la base vectorial y utiliza dicha información para construir respuestas contextualizadas.

---

## 💾 Memoria conversacional

El sistema incorpora **SQLite** para mantener memoria conversacional local.

Esto permite conservar información relacionada con la interacción del usuario con el asistente.

---

## 📚 Microaprendizaje

La interfaz de COODIBOT está diseñada para entregar el contenido de manera gradual.

En lugar de presentar inmediatamente toda la información disponible, el usuario puede interactuar con el contenido y acceder progresivamente a información adicional.

---

## 🔍 Divulgación progresiva

Los **Objetivos de Aprendizaje (OA)** pueden presentarse inicialmente de manera resumida y posteriormente desplegar sus descripciones completas.

Esto busca reducir la sobrecarga de información y facilitar la comprensión del contenido educativo.

---

# 👨‍💼 Panel de Administración

COODIBOT incorpora un panel de administración orientado a la **curación humana de información**.

Este componente implementa un enfoque **Human-in-the-loop**, permitiendo revisar las respuestas que han recibido una valoración negativa.

### Flujo de curación

```text
Usuario
   │
   ▼
Interacción con COODIBOT
   │
   ▼
Evaluación de la respuesta
   │
   ├── 👍 Positiva
   │
   └── 👎 Negativa
          │
          ▼
     Panel de Admin
          │
          ▼
   Revisión humana
          │
          ▼
  Edición de metadatos
          │
          ▼
 Actualización de Pinecone
```

Desde el panel administrativo es posible:

* Revisar respuestas reportadas con valoración negativa (`-1`).
* Editar los metadatos asociados.
* Actualizar la información almacenada en Pinecone.
* Participar directamente en el proceso de curación del conocimiento utilizado por el sistema.

El panel se encuentra disponible en:

```text
http://localhost:4200/admin
```

---

# 🛠️ Stack tecnológico

| Componente              | Tecnología        |
| ----------------------- | ----------------- |
| Frontend                | Angular           |
| Backend                 | FastAPI           |
| Lenguaje Backend        | Python            |
| Base de datos local     | SQLite            |
| Base vectorial          | Pinecone          |
| Inteligencia Artificial | OpenAI            |
| Control de versiones    | Git               |
| Arquitectura            | Monorepo          |
| Patrón de IA            | RAG               |
| Curación                | Human-in-the-loop |

---

# ⚙️ Requisitos previos

Antes de comenzar, asegúrate de tener instalado:

* **Python 3.10 o superior**
* **Node.js**
* **npm**
* **Git**

Puedes comprobar las versiones instaladas mediante:

```bash
python --version
node --version
npm --version
git --version
```

---

# 🚀 Instalación

## 1. Clonar el repositorio

Clona el repositorio y accede a la carpeta del proyecto:

```bash
git clone https://github.com/Juaquiinx/COODIBOT.git
cd COODIBOT
```

---

## 2. Configurar el Backend

Accede a la carpeta del backend:

```bash
cd backend
```

Crea un entorno virtual:

```bash
python -m venv venv
```

### Windows — PowerShell

```powershell
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Una vez activado el entorno virtual, instala las dependencias:

```bash
pip install -r requirements.txt
```

---

# 🔐 Variables de entorno

Por razones de seguridad, las credenciales utilizadas por COODIBOT **no deben almacenarse directamente en el repositorio**.

Dentro de la carpeta `backend`, crea un archivo llamado:

```text
.env
```

Utiliza como referencia el archivo `.env.example` y agrega tus credenciales:

```env
OPENAI_API_KEY=tu_llave_de_openai_aqui
PINECONE_API_KEY=tu_llave_de_pinecone_aqui
```

> ⚠️ **Importante:** nunca subas tu archivo `.env` al repositorio ni expongas tus claves privadas.

---

# 🖥️ Configuración del Frontend

Abre una nueva terminal y accede a la carpeta del frontend:

```bash
cd frontend
```

Instala las dependencias de Node:

```bash
npm install
```

---

# ▶️ Ejecución del proyecto

Para ejecutar COODIBOT en modo desarrollo es necesario mantener **dos terminales abiertas simultáneamente**.

## Terminal 1 — Backend

Accede a la carpeta `backend`, activa el entorno virtual y ejecuta:

```bash
uvicorn main:app --reload
```

El backend estará disponible en:

```text
http://127.0.0.1:8000
```

---

## Terminal 2 — Frontend

Accede a la carpeta `frontend` y ejecuta:

```bash
ng serve
```

La aplicación estará disponible en:

```text
http://localhost:4200
```

---

# 📖 Documentación de la API

FastAPI proporciona documentación interactiva de los endpoints del backend.

Una vez iniciado el servidor, puedes acceder a Swagger mediante:

```text
http://127.0.0.1:8000/docs
```

Desde allí es posible consultar y probar directamente los endpoints disponibles en la API.

También puedes acceder a la especificación OpenAPI generada por FastAPI en:

```text
http://127.0.0.1:8000/openapi.json
```

---

# 🧪 Pruebas funcionales

Una vez que ambos servicios estén ejecutándose, puedes comprobar las principales funcionalidades desde la aplicación.

### 💬 Interfaz de Chat

Desde la aplicación web:

1. Ingresa a `http://localhost:4200`.
2. Abre el botón flotante de **COODIBOT** ubicado en la esquina inferior derecha.
3. Interactúa con el asistente.
4. Evalúa las respuestas obtenidas.
5. Explora las descripciones de los Objetivos de Aprendizaje mediante divulgación progresiva.

### 👨‍💼 Panel de administración

Accede a:

```text
http://localhost:4200/admin
```

Desde esta interfaz se pueden revisar las respuestas que hayan recibido una valoración negativa (`-1`), editar sus metadatos y actualizar la información correspondiente en Pinecone.

---

# 🔄 Flujo RAG

De manera simplificada, el procesamiento de una consulta sigue el siguiente flujo:

```text
┌─────────────────────┐
│ Consulta del usuario│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Procesamiento de la │
│      consulta       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Búsqueda semántica  │
│      Pinecone       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Contexto relevante  │
│     recuperado      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Modelo de lenguaje  │
│       OpenAI        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Respuesta generada  │
│     por COODIBOT    │
└─────────────────────┘
```

---

# 📂 Estructura del proyecto

El proyecto está organizado como un monorepo:

```text
COODIBOT/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
└── README.md
```

> La estructura mostrada representa la organización general del proyecto. Los archivos y directorios adicionales pueden variar según la versión actual del repositorio.

---

# 🔒 Seguridad

Las credenciales de servicios externos deben mantenerse fuera del control de versiones.

En particular:

* No incluir `OPENAI_API_KEY` directamente en el código.
* No incluir `PINECONE_API_KEY` directamente en el código.
* Mantener las credenciales en `.env`.
* Evitar subir archivos `.env` al repositorio.
* Utilizar `.env.example` únicamente como plantilla.

---

# 🧩 Componentes del sistema

```text
COODIBOT
│
├── 🎨 Frontend
│   └── Angular
│       ├── Interfaz de chat
│       ├── Microaprendizaje
│       ├── Divulgación progresiva
│       └── Panel de administración
│
├── ⚡ Backend
│   └── FastAPI
│       └── API REST
│
├── 🧠 Inteligencia Artificial
│   └── OpenAI
│
├── 🔎 Recuperación de información
│   └── Pinecone
│       └── Base vectorial
│
└── 💾 Memoria
    └── SQLite
        └── Memoria conversacional
```

---

# 🎯 Objetivo del proyecto

COODIBOT busca facilitar el acceso de docentes de educación básica a información relacionada con la **robótica educativa**, utilizando técnicas modernas de Inteligencia Artificial y recuperación semántica.

La combinación de **RAG + modelos de lenguaje + búsqueda vectorial + memoria conversacional + curación humana** permite construir un asistente que no depende únicamente de la generación del modelo, sino que incorpora información recuperada desde una base de conocimiento especializada.

---

# 🚧 Estado del proyecto

**COODIBOT se encuentra actualmente en desarrollo.**

Las funcionalidades principales contempladas en la implementación actual incluyen:

* [x] Backend basado en FastAPI
* [x] Frontend basado en Angular
* [x] Integración con OpenAI
* [x] Integración con Pinecone
* [x] Memoria conversacional mediante SQLite
* [x] Arquitectura RAG
* [x] Sistema de valoración de respuestas
* [x] Panel de administración
* [x] Curación humana de respuestas
* [x] Actualización de información en Pinecone

---


# 👨‍💻 COODIBOT

**COODIBOT — Asistente educativo para robótica educativa**

Desarrollado como una solución basada en Inteligencia Artificial para apoyar a docentes de educación básica en Chile.

<p align="center">
  🤖 · 🧠 · 🔎 · 📚
</p>
