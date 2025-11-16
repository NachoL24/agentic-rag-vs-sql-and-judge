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
        
        prompt = f"""Eres un generador experto de SQL para una base de datos médica. 
Tu única tarea es transformar una pregunta en lenguaje natural en una consulta SQL válida.

### OBJETIVO
Dado el input del usuario, genera la consulta SQL EXACTA que responda su pregunta, sin agregar nada más.

---

### ESQUEMA (usa solo estos nombres exactos)
{schema_text}

---

### REGLAS OBLIGATORIAS
1. **Responde SOLO SQL. Nada de explicaciones, texto adicional ni formato markdown.**
2. **Nunca inventes tablas ni columnas.** Si algo no existe en el esquema, usa lo más cercano o devuelve un SELECT vacío hacia la tabla correcta.
3. Para preguntas de conteo, usa:  
   `SELECT COUNT(*) FROM tabla;`
4. Si la pregunta no menciona filtros, **no agregues WHERE**.
5. Si la pregunta menciona “todos”, “listar”, “mostrar”, usar `SELECT *`.
6. Si se mencionan condiciones, traducirlas a SQL estándar:
   - “mayores de X años” → `edad > X`
   - “menores de X años” → `edad < X`
   - “entre X e Y” → `edad BETWEEN X AND Y`
   - “con diagnóstico de X” → JOIN + condición con LIKE
7. Si la pregunta requiere unir tablas (e.g. paciente + nota médica + diagnóstico), usa JOIN correctos basados en el esquema.
8. Si la pregunta es ambigua, **elige la interpretación más literal y simple**.
9. Si la pregunta pide orden, limita y agrupa, utiliza:
   - `ORDER BY campo`
   - `GROUP BY campo`
   - `LIMIT N`
10. **NO uses comillas invertidas ni alias automáticos.**
11. Nunca agregues comentarios dentro del SQL.

---

### EJEMPLOS DE TRANSFORMACIÓN
P: "¿Cuántos pacientes existen?"  
R: `SELECT COUNT(*) FROM patients;`

P: "Listame los pacientes mayores de 60 años"  
R: `SELECT * FROM patients WHERE edad > 60;`

P: "Notas clínicas del paciente con id 5"  
R: `SELECT * FROM clinical_notes WHERE patient_id = 5;`

P: "Pacientes con diagnósticos que incluyan 'asma'"  
R: `SELECT p.* FROM patients p JOIN diagnoses d ON p.id = d.patient_id WHERE d.descripcion LIKE '%asma%';`

---

### INSTRUCCIÓN FINAL
Genera una única consulta SQL que responda la pregunta:

PREGUNTA:
{query}

SQL:
"""
        
        try:
            response = LLM.invoke([SystemMessage(content="Generas consultas SQL precisas y simples. Respondes SOLO con SQL."), HumanMessage(content=prompt)])
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

        prompt = f"""Eres un copiloto médico experto que asiste a doctores en su práctica clínica.

Consulta del médico: {query}
Datos encontrados: {results}

Como copiloto médico, proporciona:

1. **Interpretación clínica**: ¿Qué significan estos datos para la práctica médica?

2. **Consideraciones diagnósticas**: ¿Qué patologías o condiciones deberías considerar?

3. **Recomendaciones de seguimiento**: ¿Qué estudios adicionales o monitoreo sugieres?

4. **Alertas clínicas**: ¿Hay algo que requiera atención inmediata?

5. **Sugerencias de tratamiento**: ¿Qué enfoques terapéuticos podrían ser relevantes?

6. **Próximos pasos**: ¿Qué acciones concretas recomiendas?

Respuesta como copiloto médico:"""

        try:
            response = LLM.invoke([SystemMessage(content="Eres un copiloto médico que asiste a doctores con insights clínicos prácticos, diagnósticos diferenciales y recomendaciones de tratamiento. No menciones aspectos técnicos de bases de datos."), HumanMessage(content=prompt)])
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
