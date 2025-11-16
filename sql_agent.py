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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
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
        text = candidate

    # Si vino en bloque ```
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            # quitar 'sql' si está
            inner = re.sub(r"^\s*sql\s*", "", inner, flags=re.IGNORECASE).strip()
            return inner

    # Si no hay bloque, quedate con la primera línea que parezca SQL
    return text.strip()

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
      AND d.fecha >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH);

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
    HAVING COUNT(d.id) > 2;
    """).strip()

    prompt = f"""
Eres un generador de SQL para MySQL extremadamente estricto.
Tu única tarea es producir SQL válido basado EXCLUSIVAMENTE en el esquema provisto.
Responde **SIEMPRE** en este formato y termina con "SQL final:" seguido únicamente del SQL.

### ESQUEMA AUTORIZADO
{schema_text}

### REGLAS
- Prohibido inventar tablas o columnas.
- Usa SOLO tablas/columnas del esquema.
- Dialecto: MySQL. Fechas con CURDATE(), DATE_SUB, INTERVAL, etc.
- Cuando el usuario mencione una enfermedad/síntoma/condición, búscala en diagnoses.diagnostico con:
  WHERE d.diagnostico LIKE '%término%'
- Si no estás 100% seguro, responde un SQL que devuelva 0 filas pero sea sintácticamente válido.
- Solo una sentencia, sin punto y coma final, y SOLO SELECT (no DDL/DML).
- Si la pregunta pide cantidades (cuántos/cantidad/total/contar), usar COUNT(...) apropiado.
- Para "cuántos pacientes", utilizar COUNT(DISTINCT p.id) cuando corresponda.

### PASOS (escribe SIEMPRE antes del SQL final)
1) Interpretación:
2) Tablas relevantes:
3) Columnas relevantes:
4) Verificación de relaciones:
5) SQL final:

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

    def generate_sql(self, query: str) -> str:
        """
        Genera SQL con plan paso a paso + validación EXPLAIN (hasta 3 intentos).
        Requiere: self.schema y LLM. Opcional: self.db para validar con EXPLAIN.
        """
        if not hasattr(self, "schema") or not self.schema:
            return "SELECT 'Error: schema no disponible' AS error"
        if not LLM:
            return "SELECT 'Error: LLM no disponible' AS error"

        schema_text = _build_schema_text(self.schema)
        base_prompt = _build_sql_prompt(schema_text, query)

        system_msg = "Generas consultas SQL precisas y simples. Respondes SOLO con SQL al final, siguiendo el formato indicado."
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=base_prompt),
        ]

        last_error_for_model = None

        for attempt in range(3):
            if last_error_for_model:
                # Feedback loop: devolvemos el error para que el LLM corrija
                fix_msg = dedent(f"""
                El SQL previo falló con EXPLAIN por este error:

                {last_error_for_model}

                Corrige el SQL. Recuerda:
                - Solo SELECT
                - Una sola sentencia, sin ;
                - Usa exclusivamente el esquema
                - Mantén el formato con "SQL final:"
                """).strip()
                messages.append(HumanMessage(content=fix_msg))

            try:
                response = LLM.invoke(messages)
                raw = getattr(response, "content", str(response)).strip()
                sql = _extract_sql(raw)

                # Reglas locales: solo SELECT, sin múltiples sentencias
                if not _is_select_only(sql):
                    raise ValueError("La salida no es una única sentencia SELECT válida.")

                # Si la pregunta pide cantidad, reforzar COUNT(...)
                sql = _enforce_count_if_needed(query, sql).strip().rstrip(";")


                # 🔧 NUEVO: corregir alias mal calzados (p.ej. d.genero → p.genero)
                sql = _route_misqualified_columns(sql, self.schema)

                # Validación con EXPLAIN (si hay self.db)
                db = getattr(self, "db", None)
                ok, err = _validate_with_explain(db, sql)
                if ok:
                    return sql
                else:
                    last_error_for_model = err or "EXPLAIN desconocido"
                    continue

            except Exception as e:
                last_error_for_model = str(e)
                continue

        # Fallback seguro si no pudimos validar/corregir
        if last_error_for_model:
            safe_msg = last_error_for_model.replace("'", "`")
            return f"SELECT 'Error al generar SQL: {safe_msg}' AS error"
        return "SELECT 'Error inesperado al generar SQL' AS error"
    
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
        """
        Genera una opinión clínica basada en:
        - query: pregunta clínica formulada por el médico
        - sql: consulta SQL que generó los resultados (solo como contexto técnico)
        - results: resultado crudo de la consulta (lista de dicts)
        """

        if not LLM:
            # Fallback simple si no hay LLM configurado
            return f"Resultados (sin análisis clínico por falta de LLM): {results}"

        # Prompt de SISTEMA: rol del modelo
        system_prompt = """
    Eres un MÉDICO ESPECIALISTA que asiste a otros médicos interpretando datos ya analizados.

    SIEMPRE asume que tu lector ES UN PROFESIONAL DE LA SALUD, NO el paciente.

    RECIBES SIEMPRE:
    - Una PREGUNTA CLÍNICA formulada por otro médico.
    - Un BLOQUE DE DATOS CLÍNICOS (resultado crudo de un análisis).
    - OPCIONALMENTE, un BLOQUE DE CONTEXTO TÉCNICO (esquema de tablas y consulta utilizada).

    IMPORTANTE SOBRE EL CONTEXTO TÉCNICO:
    - El esquema de la base de datos, la consulta y los nombres de tablas/campos se te dan SOLO para que entiendas mejor qué representan los números (por ejemplo, que se cuentan pacientes únicos, diagnósticos, consultas, etc.).
    - NUNCA debes mencionar ni describir:
      - SQL, queries, tablas, columnas, campos, bases de datos, tipos de datos, JSON.
      - Nombres de tablas o columnas (por ejemplo, `patients`, `diagnoses`, `genero`, etc.).
    - Tu respuesta debe ser 100% clínica, como si solo hubieras recibido un resumen numérico.

    TU TAREA:
    1. Responder a la pregunta con una OPINIÓN CLÍNICA razonada, basándote en:
       - Los datos numéricos disponibles.
       - Tu conocimiento médico general.
    2. NO hablar de aspectos técnicos ni de cómo se obtuvieron los datos.

    ESTRUCTURA RECOMENDADA DE LA RESPUESTA:
    1) Resumen del hallazgo:
       - Interpreta brevemente el resultado en términos clínicos.
    2) Respuesta a la pregunta:
       - Debes responder la pregunta con una opinión clínica razonada, basándote en los datos numéricos disponibles y tu conocimiento médico general.

    REGLAS DE SEGURIDAD CLÍNICA:
    - No des diagnósticos definitivos de individuos; habla SIEMPRE en términos de la cohorte o grupo.
    - No des indicaciones directas al paciente (“usted debe…”); formula siempre sugerencias para el médico (“podría considerarse…”, “sería razonable evaluar…”).
    - No inventes números que no estén en los datos. Si necesitas cantidades, deriva solo lo que sea lógicamente inferible.

    Responde SOLO con el texto de la opinión clínica. No menciones el contexto técnico ni expliques estas instrucciones.
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

        user_prompt = f"""
    Pregunta clínica del colega:
    {query}

    Datos clínicos (resultado crudo):
    {results}

    Contexto técnico (SOLO PARA TI, NO MENCIONAR EN LA RESPUESTA):

    Schema disponible:
    {schema_text}

    SQL ejecutado:
    {sql}

    Redacta tu opinión clínica siguiendo las instrucciones del sistema.
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
            return f"Resultados (no se pudo generar análisis clínico): {results}"

    def run(self, query: str) -> str:
        # Mostrar schema disponible para debug
        if not self.schema or not self.schema.get("tables"):
            return "Error: No se pudo obtener el schema de la base de datos. Verifica la conexión."

        sql = self.generate_sql(query)
        results = self.execute_query(sql)
        analysis = self.analyze_results(query, sql, results)

        # Incluir schema en la respuesta para debug
        schema_info = "\n".join([f"{table}: {', '.join(info['columns'])}" for table, info in self.schema["tables"].items()])
        
        return f"{analysis}\n\n--- SCHEMA DISPONIBLE ---\n{schema_info}\n\n--- SQL ---\n{sql} \n--------QUERY RESULT-------------- \n{results}"

sql_agent = SQLAgent()

def run_sql_agent(query: str) -> str:
    return sql_agent.run(query)
