"""
Script para borrar la base de datos vectorial ChromaDB.
"""

import shutil
from pathlib import Path

# Configuración
SCRIPT_DIR = Path(__file__).parent
RAG_AGENT_DIR = SCRIPT_DIR / "RAG_agent"
CHROMA_DIR = RAG_AGENT_DIR / "chroma_db"


def clear_vector_database():
    """Elimina la base de datos vectorial ChromaDB"""
    
    if not CHROMA_DIR.exists():
        print(f"✓ No existe base de datos vectorial en {CHROMA_DIR}")
        print("  No hay nada que borrar.")
        return
    
    if not any(CHROMA_DIR.iterdir()):
        print(f"✓ El directorio {CHROMA_DIR} está vacío")
        print("  No hay nada que borrar.")
        return
    
    print(f"⚠️  ADVERTENCIA: Se eliminará toda la base de datos vectorial en:")
    print(f"   {CHROMA_DIR}")
    print(f"\n   Contenido actual:")
    
    # Mostrar qué hay en el directorio
    items = list(CHROMA_DIR.iterdir())
    for item in items[:10]:  # Mostrar primeros 10
        if item.is_file():
            size = item.stat().st_size
            print(f"   - {item.name} ({size:,} bytes)")
        else:
            print(f"   - {item.name}/ (directorio)")
    
    if len(items) > 10:
        print(f"   ... y {len(items) - 10} elementos más")
    
    # Pedir confirmación
    respuesta = input("\n¿Estás seguro de que quieres borrar todo? (escribe 'si' para confirmar): ").strip().lower()
    
    if respuesta in ['si', 'sí', 'yes', 'y']:
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f"\n✓ Base de datos vectorial eliminada exitosamente")
            print(f"  Directorio eliminado: {CHROMA_DIR}")
            print(f"\n  Para recrear la base de datos, ejecuta:")
            print(f"  python load_vector_data.py")
        except Exception as e:
            print(f"\n✗ Error al eliminar la base de datos: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✗ Operación cancelada. La base de datos no fue modificada.")


if __name__ == "__main__":
    clear_vector_database()

