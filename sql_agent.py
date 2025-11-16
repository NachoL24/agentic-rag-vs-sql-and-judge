import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import re
from typing import Dict, Tuple, Set, Optional
from textwrap import dedent


load_dotenv()
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:root@localhost:3306/historias_clinicas")

try:
    LLM = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
except Exception as e:
    LOG.warning(f"Error configurando Ollama: {e}")
    LLM = None

COUNT_KEYWORDS = (
    "cuántos", "cuantos", "cantidad", "número", "numero", "total", "contar", "count"
)

def _extract_sql(text: str) -> str:
    """Extrae el SQL desde bloques ``` o desde el marcador 'SQL final:'."""
    if not text:
        return ""
    text = text.strip()

    # Preferir sección 'SQL final:' si existe
    m = re.search(r"SQL\s*final\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        # cortar si el modelo pegó más comentario debajo
        candidate = candidate.split("\nInterpretación:")[0].strip()
        candidate = candidate.split("\nTablas relevantes:")[0].strip()
        candidate = candidate.split("\nColumnas relevantes:")[0].strip()
        candidate = candidate.split("\nVerificación de relaciones:")[0].strip()
        candidate = candidate.split("\nQueries necesarias:")[0].strip()
        text = candidate

    # Si vino en bloque ```
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            # quitar 'sql' si está
            inner = re.sub(r"^\s*sql\s*", "", inner, flags=re.IGNORECASE).strip()
            # Filtrar líneas que no sean SQL (explicaciones, comentarios largos)
            lines = inner.split('\n')
            sql_lines = []
            for line in lines:
                stripped = line.strip()
                # Saltar líneas vacías, comentarios, o que sean explicaciones
                if not stripped or stripped.startswith('--') or stripped.startswith('#'):
                    continue
                # Si la línea empieza con palabras clave SQL, incluirla
                if re.match(r'^\s*(SELECT|WITH|FROM|WHERE|JOIN|GROUP|HAVING|ORDER|LIMIT|UNION|AND|OR|COUNT|DISTINCT)', stripped, re.IGNORECASE):
                    sql_lines.append(line)
                # Si ya tenemos SQL y la línea parece SQL (contiene operadores o palabras clave)
                elif sql_lines and any(kw in stripped.upper() for kw in ['LIKE', '=', '(', ')', 'ON', 'AS', 'ID', 'NOMBRE', 'APELLIDO', 'GENERO', 'DIAGNOSTICO']):
                    sql_lines.append(line)
            if sql_lines:
                return '\n'.join(sql_lines)
            return inner

    # Buscar la primera línea que empiece con SELECT
    lines = text.split('\n')
    sql_lines = []
    found_select = False
    for line in lines:
        stripped = line.strip()
        # Saltar explicaciones que no sean SQL
        if not stripped or stripped.startswith('Obtener') or stripped.startswith('Contar') or stripped.startswith('Query'):
            continue
        if re.match(r'^\s*SELECT', stripped, re.IGNORECASE):
            found_select = True
            sql_lines.append(line)
        elif found_select:
            # Continuar hasta encontrar una línea que no parezca SQL
            if not stripped or stripped.startswith('--') or stripped.startswith('#'):
                break
            # Si parece SQL, agregarlo
            if any(kw in stripped.upper() for kw in ['FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'LIKE', 'COUNT', 'DISTINCT', 'GROUP', 'HAVING', 'ORDER', 'LIMIT', '=', '(', ')']):
                sql_lines.append(line)
            elif stripped and not any(word in stripped.lower() for word in ['obtener', 'contar', 'query', 'interpretación']):
                sql_lines.append(line)
            else:
                break
    
    if sql_lines:
        return '\n'.join(sql_lines)
    
    # Fallback: devolver el texto completo si no se encontró SQL claro
    return text.strip()

def _extract_multiple_sql(text: str) -> List[str]:
    """
    Extrae múltiples queries SQL desde el texto.
    Busca patrones como:
    - "Query 1:", "Query 2:", etc.
    - "SQL 1:", "SQL 2:", etc.
    - Listas numeradas de queries
    - Múltiples bloques ```sql
    """
    if not text:
        return []
    
    queries = []
    text = text.strip()
    
    # Buscar patrones de queries numeradas
    # Patrón 1: "Query 1:", "Query 2:", etc.
    query_pattern = re.compile(
        r"(?:Query|SQL)\s*(\d+)\s*:?\s*(.+?)(?=(?:Query|SQL)\s*\d+\s*:|$)",
        re.IGNORECASE | re.DOTALL
    )
    
    matches = list(query_pattern.finditer(text))
    if matches:
        for match in matches:
            query_text = match.group(2).strip()
            # Limpiar el query
            query_text = _extract_sql(query_text)
            if query_text:
                queries.append(query_text)
        return queries
    
    # Patrón 2: Lista numerada (1., 2., etc.)
    numbered_pattern = re.compile(
        r"^\s*\d+[\.\)]\s*(.+?)(?=^\s*\d+[\.\)]|$)",
        re.MULTILINE | re.DOTALL
    )
    
    matches = list(numbered_pattern.finditer(text))
    if len(matches) > 1:  # Si hay más de una, probablemente son múltiples queries
        for match in matches:
            query_text = match.group(1).strip()
            query_text = _extract_sql(query_text)
            if query_text and _is_select_only(query_text):
                queries.append(query_text)
        if queries:
            return queries
    
    # Patrón 3: Múltiples bloques ```sql
    code_blocks = re.findall(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if len(code_blocks) > 1:
        for block in code_blocks:
            query_text = re.sub(r"^\s*sql\s*", "", block, flags=re.IGNORECASE).strip()
            if query_text and _is_select_only(query_text):
                queries.append(query_text)
        if queries:
            return queries
    
    # Si no se encontraron múltiples queries, intentar extraer una sola
    single_query = _extract_sql(text)
    if single_query:
        return [single_query]
    
    return []

def _is_select_only(sql: str) -> bool:
    """Permite solo SELECT. Sin ; múltiples ni DDL/DML peligrosos."""
    s = sql.strip().rstrip(";").strip()
    # Una sola sentencia
    if ";" in s:
        return False
    # Solo SELECT (o WITH ... SELECT)
    first_token = re.match(r"^\s*([a-zA-Z]+)", s)
    if not first_token:
        return False
    tok = first_token.group(1).upper()
    if tok == "WITH":
        # Aceptar WITH... SELECT
        return bool(re.search(r"\bSELECT\b", s, flags=re.IGNORECASE))
    return tok == "SELECT"

def _enforce_count_if_needed(user_query: str, sql: str) -> str:
    """Si la pregunta pide cantidad, fuerza SELECT COUNT(...) y evita SELECT *."""
    uq = user_query.lower()
    if any(k in uq for k in COUNT_KEYWORDS):
        # Si ya hay COUNT, dejamos
        if re.search(r"\bCOUNT\s*\(", sql, flags=re.IGNORECASE):
            return sql
        # Si pregunta explícitamente por pacientes → COUNT DISTINCT p.id
        if any(k in uq for k in ["pacientes", "personas"]):
            # intentar detectar alias de patients
            # reemplazo heurístico: si hay FROM patients p → usar p.id
            m = re.search(r"from\s+(\w+)\s+([a-z]\w*)", sql, flags=re.IGNORECASE)
            count_col = "id"
            alias = None
            if m:
                from_table, alias = m.group(1), m.group(2)
                if from_table.lower() in ("patients", "pacientes"):
                    count_col = f"{alias}.id" if alias else "patients.id"
            # reemplazar SELECT ... por COUNT
            sql = re.sub(r"^(\s*SELECT\s+).+?(\s+FROM\s+)","\\1COUNT(DISTINCT " + count_col + ")\\2",
                         sql, flags=re.IGNORECASE | re.DOTALL)
            # remover GROUP BY/HAVING si quedaron colgados
            sql = re.sub(r"\bGROUP\s+BY\b.+$", "", sql, flags=re.IGNORECASE | re.DOTALL)
            sql = re.sub(r"\bHAVING\b.+$", "", sql, flags=re.IGNORECASE | re.DOTALL)
            return sql

        # Caso general: COUNT(*)
        sql = re.sub(r"^(\s*SELECT\s+).+?(\s+FROM\s+)", r"\1COUNT(*)\2",
                     sql, flags=re.IGNORECASE | re.DOTALL)
        sql = re.sub(r"\bGROUP\s+BY\b.+$", "", sql, flags=re.IGNORECASE | re.DOTALL)
        sql = re.sub(r"\bHAVING\b.+$", "", sql, flags=re.IGNORECASE | re.DOTALL)
    return sql

def _validate_with_explain(db, sql: str) -> Tuple[bool, Optional[str]]:
    """
    Intenta EXPLAIN <sql>.
    Soporta SQLAlchemy Engine o conexión DB-API.
    Si no hay db o falla el modo, retorna (True, None) para no bloquear.
    """
    if db is None:
        return True, None

    stmt = f"EXPLAIN {sql}"
    try:
        # SQLAlchemy Engine
        if hasattr(db, "connect"):
            with db.connect() as conn:
                conn.exec_driver_sql(stmt)
            return True, None
        # DB-API
        if hasattr(db, "cursor"):
            cur = db.cursor()
            cur.execute(stmt)
            _ = cur.fetchall()
            return True, None
    except Exception as e:
        return False, str(e)
    return True, None

def _build_schema_text(schema: dict) -> str:
    """
    Acepta self.schema con forma:
    {
      "tables": {
        "patients": {"columns": ["id INT", "nombre VARCHAR", ...]},
        "diagnoses": {"columns": ["id INT", "patient_id INT", "diagnostico TEXT", "fecha DATE"]}
      }
    }
    """
    parts = []
    for table, info in schema.get("tables", {}).items():
        cols = info.get("columns", [])
        parts.append(f"- {table}({', '.join(cols)})")
    return "\n".join(parts)

def _build_sql_prompt(schema_text: str, user_query: str) -> str:
    examples = dedent("""
    ### EJEMPLOS (few-shot)

    Pregunta: "Pacientes con diagnóstico de diabetes en los últimos 12 meses"
    Interpretación: Filtrar diagnósticos por texto 'diabetes' y fecha en el último año; devolver pacientes únicos.
    Tablas relevantes: patients, diagnoses
    Columnas relevantes: patients.id, patients.nombre, diagnoses.patient_id, diagnoses.diagnostico, diagnoses.fecha
    Verificación de relaciones: diagnoses.patient_id = patients.id
    SQL final:
    SELECT DISTINCT p.id, p.nombre
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.diagnostico LIKE '%diabetes%'
      AND d.fecha >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)

    Pregunta: "¿Cuántos pacientes tuvieron más de 2 diagnósticos en el último año?"
    Interpretación: Contar pacientes con COUNT(d.id) > 2 desde fecha actual - 1 año.
    Tablas relevantes: patients, diagnoses
    Columnas relevantes: patients.id, diagnoses.id, diagnoses.fecha
    Verificación de relaciones: diagnoses.patient_id = patients.id
    SQL final:
    SELECT COUNT(DISTINCT p.id)
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.fecha >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
    GROUP BY p.id
    HAVING COUNT(d.id) > 2

    Pregunta: "¿Qué pacientes tienen asma y cuántos también tienen diabetes?"
    Interpretación: Necesito dos queries: una para pacientes con asma, otra para pacientes con diabetes, y luego comparar.
    Tablas relevantes: patients, diagnoses
    Columnas relevantes: patients.id, patients.nombre, diagnoses.patient_id, diagnoses.diagnostico
    Verificación de relaciones: diagnoses.patient_id = patients.id
    Queries necesarias:
    Query 1: Obtener pacientes con asma
    Query 2: Obtener pacientes con diabetes
    SQL final:
    Query 1:
    SELECT DISTINCT p.id, p.nombre, p.apellido
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.diagnostico LIKE '%asma%'
    
    Query 2:
    SELECT DISTINCT p.id, p.nombre, p.apellido
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.diagnostico LIKE '%diabetes%'

    Pregunta: "¿Cuántos hombres tuvieron diabetes? y ¿cuántas mujeres tuvieron asma?"
    Interpretación: Necesito dos queries separadas: una para contar hombres con diabetes, otra para contar mujeres con asma.
    Tablas relevantes: patients, diagnoses
    Columnas relevantes: patients.id, patients.genero, diagnoses.patient_id, diagnoses.diagnostico
    Verificación de relaciones: diagnoses.patient_id = patients.id
    Queries necesarias:
    Query 1: Contar hombres con diabetes
    Query 2: Contar mujeres con asma
    SQL final:
    Query 1:
    SELECT COUNT(DISTINCT p.id)
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.diagnostico LIKE '%diabetes%'
      AND p.genero = 'M'
    
    Query 2:
    SELECT COUNT(DISTINCT p.id)
    FROM patients p
    JOIN diagnoses d ON d.patient_id = p.id
    WHERE d.diagnostico LIKE '%asma%'
      AND p.genero = 'F'
    """).strip()

    prompt = f"""
Eres un generador de SQL para MySQL extremadamente estricto.
Tu única tarea es producir SQL válido basado EXCLUSIVAMENTE en el esquema provisto.

IMPORTANTE: Analiza la pregunta cuidadosamente. Si necesitas múltiples queries para responder completamente, genera TODAS las queries necesarias.

### ESQUEMA AUTORIZADO
{schema_text}

### REGLAS CRÍTICAS
- Prohibido inventar tablas o columnas.
- Usa SOLO tablas/columnas del esquema.
- Dialecto: MySQL. Fechas con CURDATE(), DATE_SUB, INTERVAL, etc.
- Cuando el usuario mencione una enfermedad/síntoma/condición, búscala en diagnoses.diagnostico con:
  WHERE d.diagnostico LIKE '%término%'
- Para filtrar por género: usa p.genero = 'M' para hombres, p.genero = 'F' para mujeres.
- Si no estás 100% seguro, responde un SQL que devuelva 0 filas pero sea sintácticamente válido.
- SOLO SELECT (no DDL/DML). Sin punto y coma final en cada query.
- Si la pregunta pide cantidades (cuántos/cantidad/total/contar), usar COUNT(DISTINCT p.id) apropiado.
- IMPORTANTE: En "SQL final:" escribe SOLO el código SQL, sin explicaciones ni texto adicional.
- NO escribas "Obtener X" o explicaciones dentro del SQL. Solo código SQL puro.

### CUÁNDO GENERAR MÚLTIPLES QUERIES
Genera múltiples queries cuando:
- La pregunta requiere comparar dos o más grupos (ej: "pacientes con X y pacientes con Y")
- Necesitas datos de diferentes fuentes que no se pueden combinar en una sola query
- La pregunta tiene múltiples partes que requieren queries separadas
- Necesitas primero obtener una lista y luego filtrar o contar algo sobre esa lista

### FORMATO DE RESPUESTA

Si necesitas UNA SOLA query:
1) Interpretación:
2) Tablas relevantes:
3) Columnas relevantes:
4) Verificación de relaciones:
5) SQL final:
[tu query aquí]

Si necesitas MÚLTIPLES queries:
1) Interpretación:
2) Tablas relevantes:
3) Columnas relevantes:
4) Verificación de relaciones:
5) Queries necesarias:
   [Explica brevemente qué hace cada query]
6) SQL final:
Query 1:
[primera query]

Query 2:
[segunda query]

[etc...]

{examples}

### PREGUNTA DEL USUARIO
{user_query}
""".strip()
    return prompt


def _normalize_colname(c: str) -> str:
    # Quita tipos/ruido si vinieran ("id INT" → "id")
    c = c.strip()
    c = c.split()[0]
    c = c.rstrip(",")
    return c

def _build_column_index(schema: dict) -> Dict[str, Set[str]]:
    """
    Mapea columna -> set(tablas) en las que aparece.
    Útil para saber a qué tabla pertenece 'genero', 'nombre', etc.
    """
    col_to_tables: Dict[str, Set[str]] = {}
    for table, info in schema.get("tables", {}).items():
        for col in info.get("columns", []):
            cn = _normalize_colname(col)
            col_to_tables.setdefault(cn, set()).add(table)
    return col_to_tables

def _parse_table_aliases(sql: str) -> Dict[str, str]:
    """
    Retorna alias->tabla a partir de FROM/JOIN.
    Soporta 'FROM patients p', 'FROM patients AS p', o sin alias.
    Si no hay alias, el alias implícito es el mismo nombre de la tabla.
    """
    alias_to_table: Dict[str, str] = {}

    # FROM ...
    m = re.search(r"\bFROM\s+([a-zA-Z_]\w*)(?:\s+(?:AS\s+)?([a-zA-Z_]\w*))?", sql, flags=re.IGNORECASE)
    if m:
        tbl = m.group(1)
        alias = m.group(2) or tbl
        alias_to_table[alias] = tbl

    # JOIN ...
    for jm in re.finditer(r"\bJOIN\s+([a-zA-Z_]\w*)(?:\s+(?:AS\s+)?([a-zA-Z_]\w*))?", sql, flags=re.IGNORECASE):
        tbl = jm.group(1)
        alias = jm.group(2) or tbl
        alias_to_table[alias] = tbl

    return alias_to_table

def _route_misqualified_columns(sql: str, schema: dict) -> str:
    """
    Reescribe alias.column cuando la columna no pertenece a la tabla del alias
    y SÍ es única en otra de las tablas presentes. También califica columnas
    sin alias cuando son únicas.
    """
    alias_to_table = _parse_table_aliases(sql)
    if not alias_to_table:
        return sql  # nada que hacer

    present_tables = set(alias_to_table.values())
    col_index = _build_column_index(schema)

    # 1) Corrige 'alias.col' mal calzados
    def replace_alias_col(m):
        alias = m.group(1)
        col = m.group(2)
        # si el alias no existe, no tocamos
        if alias not in alias_to_table:
            return m.group(0)
        table_of_alias = alias_to_table[alias]

        # si la columna sí pertenece a la tabla del alias, está bien
        if col in col_index and table_of_alias in col_index[col]:
            return m.group(0)

        # Si la columna es única entre las tablas presentes, re-enrutar
        candidate_tables = col_index.get(col, set()) & present_tables
        if len(candidate_tables) == 1:
            target_table = next(iter(candidate_tables))
            # buscar el alias que corresponde a esa tabla
            target_alias = None
            for a, t in alias_to_table.items():
                if t == target_table:
                    target_alias = a
                    break
            if target_alias:
                return f"{target_alias}.{col}"

        # ambiguo o desconocido → no tocar
        return m.group(0)

    sql = re.sub(r"\b([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\b", replace_alias_col, sql)

    # 2) Califica columnas solas (sin alias) cuando son únicas
    # Evitar palabras clave SQL comunes
    keywords = {
        'select','from','where','join','on','and','or','group','by','having','order','limit',
        'count','distinct','as','inner','left','right','outer','with','case','when','then','else','end',
        'like','in','is','not','null','between','exists','union','all','any','some','top','offset','fetch'
    }

    def qualify_bare_columns(match):
        col = match.group(1)
        if col.lower() in keywords:
            return match.group(0)
        # Columnas muy comunes como 'id' suelen ser ambiguas; no calificar
        if col.lower() in ('id',):
            return match.group(0)
        candidate_tables = col_index.get(col, set()) & present_tables
        if len(candidate_tables) == 1:
            target_table = next(iter(candidate_tables))
            target_alias = None
            for a, t in alias_to_table.items():
                if t == target_table:
                    target_alias = a
                    break
            if target_alias:
                return f"{target_alias}.{col}"
        return match.group(0)

    # Sólo reescribir dentro de la cláusula WHERE/HAVING (para no romper SELECT/GROUP BY ya correctos)
    where_m = re.search(r"\bWHERE\b(.+)$", sql, flags=re.IGNORECASE | re.DOTALL)
    if where_m:
        where_clause = where_m.group(1)

        # Reescribe tokens no calificados que parezcan columnas
        # patrón: palabra que no está precedida por '.' ni por dígito/cadena
        where_clause_fixed = re.sub(
            r"(?<!\.)\b([a-zA-Z_]\w*)\b",
            qualify_bare_columns,
            where_clause
        )
        sql = sql[:where_m.start(1)] + where_clause_fixed

    return sql


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

    def generate_sql(self, query: str) -> List[str]:
        """
        Genera una o múltiples queries SQL con plan paso a paso + validación EXPLAIN (hasta 3 intentos).
        Retorna una lista de queries SQL.
        Requiere: self.schema y LLM. Opcional: self.db para validar con EXPLAIN.
        """
        if not hasattr(self, "schema") or not self.schema:
            return ["SELECT 'Error: schema no disponible' AS error"]
        if not LLM:
            return ["SELECT 'Error: LLM no disponible' AS error"]

        schema_text = _build_schema_text(self.schema)
        base_prompt = _build_sql_prompt(schema_text, query)

        system_msg = "Generas consultas SQL precisas y simples. Puedes generar una o múltiples queries según sea necesario. Responde siguiendo el formato indicado."
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=base_prompt),
        ]

        last_error_for_model = None

        for attempt in range(3):
            if last_error_for_model:
                # Feedback loop: devolvemos el error para que el LLM corrija
                fix_msg = dedent(f"""
                Las queries previas fallaron con EXPLAIN por este error:

                {last_error_for_model}

                Corrige las queries. Recuerda:
                - Solo SELECT
                - Sin punto y coma final en cada query
                - Usa exclusivamente el esquema
                - Si generas múltiples queries, usa el formato "Query 1:", "Query 2:", etc.
                """).strip()
                messages.append(HumanMessage(content=fix_msg))

            try:
                response = LLM.invoke(messages)
                raw = getattr(response, "content", str(response)).strip()
                
                # Intentar extraer múltiples queries
                sql_queries = _extract_multiple_sql(raw)
                
                if not sql_queries:
                    raise ValueError("No se pudieron extraer queries válidas de la respuesta")

                # Validar y corregir cada query
                validated_queries = []
                for sql in sql_queries:
                    # Reglas locales: solo SELECT
                    if not _is_select_only(sql):
                        raise ValueError(f"La query no es una sentencia SELECT válida: {sql[:50]}...")

                    # Si la pregunta pide cantidad, reforzar COUNT(...)
                    sql = _enforce_count_if_needed(query, sql).strip().rstrip(";")

                    # Corregir alias mal calzados (p.ej. d.genero → p.genero)
                    sql = _route_misqualified_columns(sql, self.schema)
                    
                    validated_queries.append(sql)

                # Validación con EXPLAIN (si hay self.db) - validar todas las queries
                db = getattr(self, "db", None)
                all_valid = True
                first_error = None
                
                for sql in validated_queries:
                    ok, err = _validate_with_explain(db, sql)
                    if not ok:
                        all_valid = False
                        if not first_error:
                            first_error = err or "EXPLAIN desconocido"
                
                if all_valid:
                    return validated_queries
                else:
                    last_error_for_model = first_error
                    continue

            except Exception as e:
                last_error_for_model = str(e)
                continue

        # Fallback seguro si no pudimos validar/corregir
        if last_error_for_model:
            safe_msg = last_error_for_model.replace("'", "`")
            return [f"SELECT 'Error al generar SQL: {safe_msg}' AS error"]
        return ["SELECT 'Error inesperado al generar SQL' AS error"]
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Ejecuta una sola query SQL y retorna los resultados."""
        if not self.engine:
            return [{"error": "No hay conexión a BD"}]
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as e:
            LOG.error(f"Error ejecutando SQL: {e}")
            return [{"error": str(e)}]
    
    def execute_queries(self, sql_queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Ejecuta múltiples queries SQL y retorna una lista de resultados.
        Cada elemento de la lista corresponde a los resultados de una query.
        """
        if not self.engine:
            return [[{"error": "No hay conexión a BD"}]]
        
        all_results = []
        for i, sql in enumerate(sql_queries):
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql))
                    query_results = [dict(row._mapping) for row in result]
                    all_results.append(query_results)
                    LOG.info(f"Query {i+1}/{len(sql_queries)} ejecutada exitosamente. Resultados: {len(query_results)} filas")
            except SQLAlchemyError as e:
                LOG.error(f"Error ejecutando query {i+1}: {e}")
                all_results.append([{"error": f"Query {i+1}: {str(e)}"}])
        
        return all_results

    def analyze_results(self, query: str, sql_queries: List[str], all_results: List[List[Dict[str, Any]]]) -> str:
        """
        Genera una opinión clínica basada en:
        - query: pregunta clínica formulada por el médico
        - sql_queries: lista de consultas SQL que generaron los resultados (solo como contexto técnico)
        - all_results: lista de resultados, donde cada elemento es el resultado de una query (lista de dicts)
        """

        if not LLM:
            # Fallback simple si no hay LLM configurado
            return f"Resultados (sin análisis clínico por falta de LLM): {all_results}"

        # Prompt de SISTEMA: rol del modelo - VERSIÓN CONCISA Y DIRECTA
        system_prompt = """
Eres un MÉDICO ESPECIALISTA que responde preguntas clínicas de forma DIRECTA y CONCISA.

REGLAS ESTRICTAS:
1. Responde DIRECTAMENTE la pregunta del colega médico, sin rodeos ni repeticiones.
2. Si la pregunta tiene múltiples partes (ej: "X y Y"), responde a TODAS las partes.
3. Si la pregunta pide una lista de pacientes, lista SOLO los nombres completos (nombre y apellido) sin repeticiones.
4. Si la pregunta pide una cantidad, da el número exacto que aparece en los datos.
5. NO repitas información. NO divagues. NO menciones aspectos técnicos (SQL, tablas, bases de datos).
6. NO inventes información. Si los datos muestran un error o están vacíos, di claramente "No se pudo obtener la información solicitada" o "No se encontraron datos".
7. Si los datos contienen un error (ej: {'error': '...'}), NO inventes números. Di que hubo un error al obtener los datos.
8. Sé breve y preciso. Una respuesta clara y directa es mejor que una larga y confusa.

FORMATO:
- Para listas de pacientes: "Los pacientes son: [Nombre Apellido], [Nombre Apellido], ..."
- Para cantidades: "Se identifican X pacientes..." o "X hombres..." / "Y mujeres..."
- Para preguntas con múltiples partes: Responde cada parte claramente, ej: "Hombres con diabetes: X. Mujeres con asma: Y."
- Si hay error en los datos: "No se pudo obtener la información solicitada debido a un error en la consulta."

El contexto técnico (schema, SQL) se proporciona solo para que entiendas los datos, pero NUNCA debes mencionarlo en tu respuesta.
""".strip()

        # Armamos el prompt de usuario con:
        # - pregunta clínica
        # - datos crudos
        # - contexto técnico (schema + SQL) marcado como NO mencionable
        schema_text = ""
        try:
            # Si tu schema es un dict/list ["table: cols..."], lo convertimos a texto
            if self.schema:
                if isinstance(self.schema, (list, tuple)):
                    schema_text = "\n".join(self.schema)
                else:
                    schema_text = str(self.schema)
        except AttributeError:
            schema_text = "No schema disponible"

        # Formatear múltiples resultados
        results_text = ""
        if len(all_results) == 1:
            results_text = str(all_results[0])
        else:
            results_text = "\n\n".join([
                f"Resultados de Query {i+1}:\n{str(results)}"
                for i, results in enumerate(all_results)
            ])
        
        # Formatear múltiples queries SQL
        sql_text = ""
        if len(sql_queries) == 1:
            sql_text = sql_queries[0]
        else:
            sql_text = "\n\n".join([
                f"Query {i+1}:\n{sql}"
                for i, sql in enumerate(sql_queries)
            ])

        # Verificar si hay errores en los resultados
        has_errors = False
        for results in all_results:
            if results and isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and 'error' in results[0]:
                    has_errors = True
                    break

        error_instruction = ""
        if has_errors:
            error_instruction = "\n\n⚠️ IMPORTANTE: Los datos contienen errores. NO inventes números. Di claramente que hubo un error al obtener los datos."

        user_prompt = f"""
Pregunta clínica del colega:
{query}

Datos clínicos (resultado crudo):
{results_text}
{error_instruction}

Contexto técnico (SOLO PARA TI, NO MENCIONAR EN LA RESPUESTA):
Schema: {schema_text}
SQL: {sql_text}

INSTRUCCIÓN: 
- Responde DIRECTAMENTE la pregunta del colega. 
- Si la pregunta tiene múltiples partes, responde a TODAS.
- Si hay errores en los datos, NO inventes números. Di que hubo un error.
- Sé CONCISO. NO repitas información. NO divagues.
""".strip()

        try:
            response = LLM.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            # Algunos LLMs devuelven .content directamente, otros en .content[0].text; ajusta si hace falta
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return content.strip()
            # fallback por si el objeto viene más raro
            return str(response).strip()
        except Exception:
            # En caso de error, devolvemos al menos los resultados crudos
            return f"Resultados (no se pudo generar análisis clínico): {all_results}"

    def run(self, query: str) -> str:
        # Mostrar schema disponible para debug
        if not self.schema or not self.schema.get("tables"):
            return "Error: No se pudo obtener el schema de la base de datos. Verifica la conexión."

        # Generar una o múltiples queries
        sql_queries = self.generate_sql(query)
        
        # Ejecutar todas las queries
        all_results = self.execute_queries(sql_queries)
        
        # Analizar todos los resultados juntos
        analysis = self.analyze_results(query, sql_queries, all_results)

        # Incluir schema en la respuesta para debug
        schema_info = "\n".join([f"{table}: {', '.join(info['columns'])}" for table, info in self.schema["tables"].items()])
        
        # Formatear queries y resultados para la salida
        if len(sql_queries) == 1:
            sql_display = sql_queries[0]
            results_display = all_results[0]
        else:
            sql_display = "\n\n".join([
                f"Query {i+1}:\n{sql}"
                for i, sql in enumerate(sql_queries)
            ])
            results_display = "\n\n".join([
                f"Resultados Query {i+1}:\n{results}"
                for i, results in enumerate(all_results)
            ])
        
        return f"{analysis}\n\n--- SCHEMA DISPONIBLE ---\n{schema_info}\n\n--- SQL ---\n{sql_display} \n--------QUERY RESULT-------------- \n{results_display}"

sql_agent = SQLAgent()

def run_sql_agent(query: str) -> str:
    return sql_agent.run(query)
