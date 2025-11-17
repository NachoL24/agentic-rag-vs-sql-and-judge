"""
Script para mostrar todos los datos de la base de datos vectorial ChromaDB.
"""

import os
import json
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
SCRIPT_DIR = Path(__file__).parent
RAG_AGENT_DIR = SCRIPT_DIR / "RAG_agent"
CHROMA_DIR = RAG_AGENT_DIR / "chroma_db"
COLLECTION_NAME = "historias_clinicas"
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def show_all_vector_data():
    """Muestra todos los datos almacenados en la base de datos vectorial"""
    
    # Verificar que existe la base de datos
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"⚠️  No se encontró base de datos vectorial en {CHROMA_DIR}")
        print("   Ejecuta primero: python load_vector_data.py")
        return
    
    print("📊 Conectando a la base de datos vectorial...")
    print(f"   Directorio: {CHROMA_DIR}")
    print(f"   Colección: {COLLECTION_NAME}")
    print()
    
    try:
        # Conectar a la base de datos
        embeddings = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_HOST,
        )
        
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        
        # Obtener todos los documentos usando get()
        print("🔍 Obteniendo todos los documentos...")
        all_data = vectordb.get()
        
        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        total = len(ids)
        print(f"✓ Total de documentos encontrados: {total}\n")
        
        if total == 0:
            print("   La base de datos está vacía.")
            return
        
        # Mostrar cada documento
        print("=" * 80)
        for i, (doc_id, doc_content, metadata) in enumerate(zip(ids, documents, metadatas), 1):
            print(f"\n📄 DOCUMENTO {i}/{total}")
            print("-" * 80)
            print(f"ID: {doc_id}")
            print(f"\nMETADATOS:")
            for key, value in metadata.items():
                print(f"  • {key}: {value}")
            
            print(f"\nCONTENIDO:")
            # Mostrar el contenido con un límite de caracteres para legibilidad
            content_preview = doc_content[:500] if len(doc_content) > 500 else doc_content
            print(f"  {content_preview}")
            if len(doc_content) > 500:
                print(f"  ... (truncado, {len(doc_content)} caracteres en total)")
            
            print("=" * 80)
        
        # Resumen estadístico
        print("\n📈 RESUMEN ESTADÍSTICO")
        print("-" * 80)
        print(f"Total de documentos: {total}")
        
        # Contar por paciente
        if metadatas:
            patient_counts = {}
            for meta in metadatas:
                patient_id = meta.get("patient_id", "unknown")
                patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
            
            print(f"\nDocumentos por paciente:")
            for patient_id, count in sorted(patient_counts.items()):
                print(f"  • Paciente {patient_id}: {count} documento(s)")
        
        # Contar por sección si existe
        if metadatas:
            section_counts = {}
            for meta in metadatas:
                section = meta.get("seccion", "sin_seccion")
                section_counts[section] = section_counts.get(section, 0) + 1
            
            if any(s != "sin_seccion" for s in section_counts.keys()):
                print(f"\nDocumentos por sección:")
                for section, count in sorted(section_counts.items()):
                    print(f"  • {section}: {count} documento(s)")
        
        # Estadísticas de contenido
        total_chars = sum(len(doc) for doc in documents)
        avg_chars = total_chars / total if total > 0 else 0
        print(f"\nEstadísticas de contenido:")
        print(f"  • Total de caracteres: {total_chars:,}")
        print(f"  • Promedio de caracteres por documento: {avg_chars:.1f}")
        print(f"  • Documento más largo: {max(len(doc) for doc in documents) if documents else 0:,} caracteres")
        print(f"  • Documento más corto: {min(len(doc) for doc in documents) if documents else 0:,} caracteres")
        
    except Exception as e:
        print(f"\n✗ Error al acceder a la base de datos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    show_all_vector_data()

