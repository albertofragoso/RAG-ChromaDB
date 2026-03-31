# RAG API - Café Aurora ☕️🤖

Este proyecto es una API de **Generación Aumentada por Recuperación (RAG)** diseñada para responder preguntas basadas en documentos específicos de "Café Aurora". Utiliza **FastAPI** para la interfaz, **ChromaDB** como base de datos vectorial y **LangChain** para orquestar la lógica de recuperación y generación.

## 🚀 Características Principales

- **Gestión de Colecciones:** Crea, lista y elimina colecciones en ChromaDB de forma dinámica.
- **Ingesta Inteligente:** Carga de documentos con segmentación (chunking) automática.
- **Consultas Stateless:** Endpoint para preguntas rápidas y directas sobre una colección.
- **Chat con Memoria:** Historial de conversación persistente por `session_id`.
- **Estadísticas:** Análisis de fuentes y categorías dentro de cada colección.
- **Script CLI:** Herramienta interactiva (`rag_chroma.py`) para pruebas rápidas localmente.

## 🛠️ Tecnologías

- **Framework:** FastAPI
- **Base de Datos Vectorial:** ChromaDB
- **LLM:** OpenAI GPT-4o-mini (configurable)
- **Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Orquestación:** LangChain

## 📦 Instalación y Configuración

### 1. Requisitos Previos
- Python 3.10 o superior.
- Una cuenta de OpenAI (para el LLM).

### 2. Clonar y Preparar Entorno
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:
```env
OPENAI_API_KEY=tu_api_key_aqui
MODEL_NAME=gpt-4o-mini
# Opcional: si usas un proxy o gateway
# OPENAI_BASE_URL=https://tu-url-personalizada
```

## 🎮 Uso

### Iniciar el Servidor API
```bash
uvicorn main:app --reload
```
La documentación interactiva estará disponible en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Script Interactivo (CLI)
Para probar el sistema RAG directamente desde la terminal:
```bash
python rag_chroma.py
```
*Asegúrate de colocar tus archivos `.txt` en la carpeta `./docs` antes de ejecutar el script.*

## 📂 Estructura del Proyecto

- `main.py`: Punto de entrada de la API FastAPI y definición de endpoints.
- `rag_chroma.py`: Script interactivo para orquestación RAG local.
- `chroma_db/`: Directorio donde se almacenan los datos persistentes de la base vectorial.
- `docs/`: Carpeta para documentos fuente (TXT).
- `requirements.txt`: Lista de dependencias del proyecto.

## 📝 API Endpoints (Resumen)

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Estado de salud de la API. |
| `POST` | `/collections` | Crea una nueva colección. |
| `GET` | `/collections` | Lista todas las colecciones. |
| `POST` | `/collections/{name}/documents` | Agrega documentos a una colección. |
| `POST` | `/collections/{name}/query` | Pregunta puntual (sin memoria). |
| `POST` | `/collections/{name}/chat` | Conversación fluida con historial. |
| `GET` | `/collections/{name}/stats` | Estadísticas de la colección. |
