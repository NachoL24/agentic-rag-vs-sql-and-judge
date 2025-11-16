import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:root@localhost:3306/historias_clinicas")

try:
    LLM = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
except Exception as e:
    LOG.warning(f"Error configurando Ollama: {e}")
    LLM = None

class SQLAgent:
    def __init__(self):
        try:
            self.engine = create_engine(DB_URL)
            self.schema = self.get_schema()
        except Exception as e:
            LOG.error(f"Error conectando a BD: {e}")
            self.engine = None
            self.schema = {}
    
    def get_schema(self) -> Dict[str, Any]:
        if not self.engine:
            return {}
        
        try:
            inspector = inspect(self.engine)
            tables = {}
            for table_name in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns(table_name)]
                tables[table_name] = {"columns": columns}
            return {"tables": tables}
        except Exception as e:
            LOG.error(f"Error obteniendo schema: {e}")
            return {}
    
    def generate_sql(self, query: str) -> str:
        if not LLM or not self.schema:
            return "SELECT 'Error: LLM o schema no disponible';"
        
        schema_text = "\n".join([f"{table}: {', '.join(info['columns'])}" for table, info in self.schema["tables"].items()])
        
        prompt = f"""Eres un experto en SQL médico. Genera la consulta SQL más precisa para: {query}

Esquema de la base de datos (USA EXACTAMENTE estos nombres):
{schema_text}

Reglas importantes:
1. Usa EXACTAMENTE los nombres de tablas y columnas del esquema
2. Si preguntan por "pacientes", usa la tabla "patients"
3. Si preguntan por "diagnósticos", usa la tabla "diagnoses" 
4. Si preguntan por "notas clínicas", usa la tabla "clinical_notes"
5. Piensa qué tabla responde mejor a la pregunta
6. Usa JOINs cuando sea necesario para obtener información completa

SQL:"""
        
        try:
            response = LLM.invoke([SystemMessage(content="Generas SQL preciso."), HumanMessage(content=prompt)])
            sql = response.content.strip()
            if "```" in sql:
                sql = sql.split("```")[1].replace("sql", "").strip()
            return sql
        except Exception as e:
            return f"SELECT 'Error: {e}';"
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        if not self.engine:
            return [{"error": "No hay conexión a BD"}]
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as e:
            LOG.error(f"Error ejecutando SQL: {e}")
            return [{"error": str(e)}]
    
    def analyze_results(self, query: str, sql: str, results: List[Dict]) -> str:
        if not LLM:
            return f"Resultados: {results}"

        prompt = f"""Eres un analista médico experto. Analiza estos resultados y proporciona un análisis médico completo:

Pregunta original: {query}
Consulta SQL ejecutada: {sql}
Resultados obtenidos: {results}

Proporciona un análisis médico que incluya:
1. Interpretación directa de los datos
2. Significado clínico de los números
3. Posibles implicaciones epidemiológicas
4. Recomendaciones para el seguimiento médico
5. Consideraciones sobre la calidad de los datos
6. Sugerencias para análisis adicionales

Análisis médico completo:"""

        try:
            response = LLM.invoke([SystemMessage(content="Eres un analista médico experto que proporciona análisis detallados y recomendaciones clínicas basadas en datos."), HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Análisis: {results}"

    def run(self, query: str) -> str:
        # Mostrar schema disponible para debug
        if not self.schema or not self.schema.get("tables"):
            return "Error: No se pudo obtener el schema de la base de datos. Verifica la conexión."

        sql = self.generate_sql(query)
        results = self.execute_query(sql)
        analysis = self.analyze_results(query, sql, results)

        # Incluir schema en la respuesta para debug
        schema_info = "\n".join([f"{table}: {', '.join(info['columns'])}" for table, info in self.schema["tables"].items()])
        
        return f"{analysis}\n\n--- SCHEMA DISPONIBLE ---\n{schema_info}\n\n--- SQL ---\n{sql}"

sql_agent = SQLAgent()

def run_sql_agent(query: str) -> str:
    return sql_agent.run(query)
