#!/usr/bin/env python3
"""
Script para crear el schema de la base de datos y ejecutar los datos de seed.
Conecta a MySQL en Docker Compose y crea las tablas necesarias.
"""

import os
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Configuración de la base de datos (desde docker-compose.yaml)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "historias_clinicas")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Definición del schema
SCHEMA_SQL = """
-- Tabla de pacientes
CREATE TABLE IF NOT EXISTS patients (
    id INT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    apellido VARCHAR(100) NOT NULL,
    edad INT NOT NULL,
    genero CHAR(1) NOT NULL,
    INDEX idx_nombre (nombre),
    INDEX idx_apellido (apellido)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de notas clínicas
CREATE TABLE IF NOT EXISTS clinical_notes (
    id INT PRIMARY KEY,
    patient_id INT NOT NULL,
    texto TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de diagnósticos
CREATE TABLE IF NOT EXISTS diagnoses (
    id INT PRIMARY KEY,
    patient_id INT NOT NULL,
    diagnostico VARCHAR(200) NOT NULL,
    fecha DATE NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_diagnostico (diagnostico),
    INDEX idx_fecha (fecha)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def read_seed_file(seed_file_path: str) -> str:
    """Lee el archivo SQL de seed."""
    try:
        with open(seed_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        LOG.error(f"Archivo no encontrado: {seed_file_path}")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"Error leyendo archivo seed: {e}")
        sys.exit(1)


def execute_sql_statements(engine, sql_content: str, description: str):
    """Ejecuta múltiples sentencias SQL separadas por punto y coma."""
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    with engine.connect() as conn:
        for i, statement in enumerate(statements, 1):
            if not statement:
                continue
            try:
                conn.execute(text(statement))
                LOG.info(f"{description} - Sentencia {i}/{len(statements)} ejecutada correctamente")
            except SQLAlchemyError as e:
                # Ignorar errores de "ya existe" para CREATE TABLE IF NOT EXISTS
                if "already exists" in str(e).lower() or "duplicate entry" in str(e).lower():
                    LOG.warning(f"{description} - Sentencia {i}: {str(e)[:100]}")
                else:
                    LOG.error(f"{description} - Error en sentencia {i}: {e}")
                    raise
        conn.commit()
        LOG.info(f"{description} - Todas las sentencias ejecutadas correctamente")


def check_tables_exist(engine) -> bool:
    """Verifica si las tablas ya existen."""
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = ['patients', 'clinical_notes', 'diagnoses']
        return all(table in existing_tables for table in required_tables)
    except Exception as e:
        LOG.warning(f"Error verificando tablas existentes: {e}")
        return False


def drop_tables_if_exists(engine):
    """Elimina las tablas si existen (en orden inverso por foreign keys)."""
    drop_order = ['diagnoses', 'clinical_notes', 'patients']
    with engine.connect() as conn:
        for table in drop_order:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                LOG.info(f"Tabla {table} eliminada (si existía)")
            except SQLAlchemyError as e:
                LOG.warning(f"Error eliminando tabla {table}: {e}")
        conn.commit()


def main():
    """Función principal."""
    LOG.info("=" * 60)
    LOG.info("Script de seed de base de datos")
    LOG.info("=" * 60)
    
    # Obtener ruta del archivo seed
    script_dir = Path(__file__).parent
    seed_file = script_dir / "seeds" / "seed.sql"
    
    if not seed_file.exists():
        LOG.error(f"Archivo seed no encontrado: {seed_file}")
        sys.exit(1)
    
    # Crear engine
    try:
        LOG.info(f"Conectando a base de datos: {DB_NAME} en {DB_HOST}:{DB_PORT}")
        engine = create_engine(DB_URL, echo=False)
        
        # Verificar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        LOG.info("Conexión exitosa a la base de datos")
    except SQLAlchemyError as e:
        LOG.error(f"Error conectando a la base de datos: {e}")
        LOG.error("Asegúrate de que Docker Compose esté ejecutándose y MySQL esté listo")
        sys.exit(1)
    
    # Verificar si las tablas ya existen
    tables_exist = check_tables_exist(engine)
    
    if tables_exist:
        LOG.warning("Las tablas ya existen en la base de datos")
        response = input("¿Deseas eliminar las tablas existentes y recrearlas? (s/N): ")
        if response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
            LOG.info("Eliminando tablas existentes...")
            drop_tables_if_exists(engine)
        else:
            LOG.info("Manteniendo tablas existentes. Solo se insertarán datos nuevos.")
    
    # Crear schema
    LOG.info("Creando schema de tablas...")
    try:
        execute_sql_statements(engine, SCHEMA_SQL, "Creación de schema")
        LOG.info("Schema creado correctamente")
    except Exception as e:
        LOG.error(f"Error creando schema: {e}")
        sys.exit(1)
    
    # Leer y ejecutar seed file
    LOG.info(f"Leyendo archivo seed: {seed_file}")
    seed_content = read_seed_file(str(seed_file))
    
    LOG.info("Ejecutando sentencias INSERT del archivo seed...")
    try:
        execute_sql_statements(engine, seed_content, "Inserción de datos")
        LOG.info("Datos insertados correctamente")
    except Exception as e:
        LOG.error(f"Error insertando datos: {e}")
        sys.exit(1)
    
    # Verificar datos insertados
    LOG.info("Verificando datos insertados...")
    try:
        with engine.connect() as conn:
            patients_count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
            notes_count = conn.execute(text("SELECT COUNT(*) FROM clinical_notes")).scalar()
            diagnoses_count = conn.execute(text("SELECT COUNT(*) FROM diagnoses")).scalar()
            
            LOG.info("=" * 60)
            LOG.info("Resumen de datos insertados:")
            LOG.info(f"  - Pacientes: {patients_count}")
            LOG.info(f"  - Notas clínicas: {notes_count}")
            LOG.info(f"  - Diagnósticos: {diagnoses_count}")
            LOG.info("=" * 60)
    except Exception as e:
        LOG.warning(f"Error verificando datos: {e}")
    
    LOG.info("¡Proceso completado exitosamente!")


if __name__ == "__main__":
    main()

