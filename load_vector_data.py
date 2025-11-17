"""
Script para cargar los datos de vector_seed.json en la base de datos vectorial Chroma.
"""

import json
import os
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuración - usar rutas relativas al directorio del script
SCRIPT_DIR = Path(__file__).parent
SEED_FILE = SCRIPT_DIR / "seeds" / "vector_seed.json"
RAG_AGENT_DIR = SCRIPT_DIR / "RAG_agent"
CHROMA_DIR = RAG_AGENT_DIR / "chroma_db"
COLLECTION_NAME = "historias_clinicas"
EMBEDDINGS_MODEL = "nomic-embed-text"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


def load_vector_seed():
    """Carga los datos del archivo vector_seed.json"""
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo {SEED_FILE}")
    
    with open(SEED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_documents_from_seed(seed_data):
    """Convierte los datos del seed en documentos de LangChain"""
    documents = []
    
    for item in seed_data:
        # El campo puede ser 'chunk' o 'text' dependiendo del formato
        chunk = item.get('chunk') or item.get('text', '')
        patient_id = item.get('patient_id', 'unknown')
        item_type = item.get('type', 'unknown')
        item_id = item.get('id', '')
        
        # Obtener metadatos anidados si existen
        nested_metadata = item.get('metadata', {})
        
        # Crear un documento con metadata completa
        doc = Document(
            page_content=chunk,
            metadata={
                "source": f"patient_{patient_id}",
                "patient_id": patient_id,
                "type": item_type,
                "id": item_id,
                **nested_metadata  # Incluir todos los metadatos anidados
            }
        )
        documents.append(doc)
    
    return documents


def build_vector_store_from_seed():
    """Construye la base de datos vectorial desde el seed JSON"""
    print("Cargando datos del seed...")
    seed_data = load_vector_seed()
    print(f"   -> {len(seed_data)} registros encontrados")
    
    print("Creando documentos...")
    documents = create_documents_from_seed(seed_data)
    print(f"   -> {len(documents)} documentos creados")
    
    print("Creando embeddings...")
    print(f"   -> Modelo: {EMBEDDINGS_MODEL}")
    print(f"   -> Host: {OLLAMA_HOST}")
    embeddings = OllamaEmbeddings(
        model=EMBEDDINGS_MODEL,
        base_url=OLLAMA_HOST,
    )
    
    # Asegurarse de que el directorio existe
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Si ya existe una base de datos, eliminarla primero para recrearla
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"   -> Base de datos existente encontrada en {CHROMA_DIR}")
        print("   -> Se recreará con los nuevos datos...")
    
    print("Creando base de datos vectorial...")
    print("   -> Esto puede tardar unos minutos...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    
    print(f"\n✓ Base de datos vectorial construida y persistida en {CHROMA_DIR}")
    print(f"✓ Colección: {COLLECTION_NAME}")
    print(f"✓ Total de documentos: {len(documents)}")
    
    return vectordb


if __name__ == "__main__":
    try:
        build_vector_store_from_seed()
        print("\n✓ Proceso completado exitosamente")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

