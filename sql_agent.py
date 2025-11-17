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
import json  # NUEVO



load_dotenv()
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20B")
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

    IMPORTANTE:
    - Escapamos '%' -> '%%' para evitar que el driver DB-API (pymysql)
      intente interpretar los '%' de los LIKE como placeholders de formato.
    """
    if db is None:
        return True, None

    # Escapar '%' para que el driver no intente hacer interpolación de formato
    safe_sql = sql.replace("%", "%%")
    stmt = f"EXPLAIN {safe_sql}"

    try:
        # SQLAlchemy Engine
        if hasattr(db, "connect"):
            with db.connect() as conn:
                conn.exec_driver_sql(stmt)
            return True, None
        # DB-API directo
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
    prompt = f"""
Eres un generador de SQL para MySQL extremadamente estricto.
Tu única tarea es producir UNA sola sentencia SELECT válida basada EXCLUSIVAMENTE en el esquema provisto.

### ESQUEMA AUTORIZADO
{schema_text}

### REGLAS GENERALES (OBLIGATORIAS)
- Prohibido inventar tablas o columnas.
- Usa SOLO tablas y columnas que aparezcan en el esquema autorizado.
- Dialecto: MySQL. Puedes usar funciones estándar (CURDATE(), DATE_SUB, INTERVAL, etc.) si son necesarias.
- La sentencia DEBE ser únicamente un SELECT (o WITH ... SELECT).
- No incluyas comentarios, ni explicaciones, ni código adicional.
- No uses punto y coma final.
- No generes más de una sentencia.

### REGLAS PARA ENFERMEDADES / DIAGNÓSTICOS
- Cuando el usuario mencione una enfermedad, síntoma o condición (por ejemplo "diabetes", "asma"):
  - Debes filtrarla utilizando la columna diagnoses.diagnostico con un patrón LIKE:
    WHERE d.diagnostico LIKE '%término%'

### REGLAS ESPECÍFICAS PARA GÉNERO
- La ÚNICA columna para género/sexo en este esquema es 'genero' de la tabla 'patients'.
- NO inventes columnas como 'gender', 'sexo' u otras.
- Cuando el usuario hable de:
  - "hombres", "varones" → filtra con p.genero = 'M'
  - "mujeres" → filtra con p.genero = 'F'

### REGLAS PARA CANTIDADES
- Si la pregunta pide cantidades (cuántos/cantidad/total/contar):
  - Para "¿cuántos pacientes...?" utiliza COUNT(DISTINCT p.id) siempre que estés contando pacientes.
  - No calcules proporciones, porcentajes ni promedios a menos que el usuario los pida explícitamente.
- Si hay duda, devuleve un SELECT que sea sintácticamente válido, aunque devuelva 0 filas.

### FORMATO DE RESPUESTA (ESTRICTO)
Debes responder SIEMPRE en este formato de texto plano:

Interpretación: <breve explicación en una sola línea>
Tablas relevantes: <lista de tablas usadas>
Columnas relevantes: <lista de columnas usadas>
Verificación de relaciones: <explicación breve de los JOINs>
SQL final: <AQUÍ SOLO LA SENTENCIA SELECT SIN PUNTO Y COMA>

No añadas nada más antes o después.

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

    def plan_data_requirements(self, query: str) -> Dict[str, Any]:
        """
        Usa el LLM para desglosar la pregunta clínica en varias subconsultas de datos.
        Devuelve un dict con formato:
        {
          "subqueries": [
            {
              "id": "string",
              "description": "para qué sirve esta subconsulta",
              "question": "pregunta en lenguaje natural para generar SQL"
            },
            ...
          ]
        }
        Si falla, devuelve un plan mínimo con solo la pregunta original.
        """
        if not LLM:
            # Sin LLM → plan trivial
            return {
                "subqueries": [
                    {
                        "id": "main",
                        "description": "Consulta principal",
                        "question": query,
                    }
                ]
            }

        schema_text = _build_schema_text(self.schema)

        system_prompt = (
            "Eres un médico experto y analista de datos clínicos. "
            "Tu única tarea es DESGLOSAR una pregunta clínica en una o varias subconsultas de datos, "
            "SIN cambiar el sentido clínico de la pregunta original. "
            "NO generas SQL, solo un plan estructurado."
        )

        user_prompt = f"""
Pregunta clínica del colega:
{query}

Esquema de la base de datos (para que sepas qué datos existen):
{schema_text}

REGLAS CLÍNICAS Y DE DESGLOSE (OBLIGATORIAS):
- NO cambies la intención de la pregunta original.
  - Si el colega pide "¿cuántos...?", las subconsultas también deben ser de tipo "¿cuántos...?".
  - NO reemplaces "¿cuántos...?" por "¿qué proporción...?" ni por "¿cuál es el promedio...?".
- Si el colega hace varias preguntas en la misma frase, genera UNA subconsulta independiente por cada pregunta o métrica explícita.
- Si se mencionan enfermedades (por ejemplo "diabetes", "asma", etc.), deben quedar explícitas en el campo 'question' de cada subconsulta.
- Si se mencionan hombres/mujeres:
  - Asume que el sexo/género proviene de la columna 'genero' de la tabla de pacientes, con valores 'M' y 'F'.
- No inventes métricas nuevas que el colega no haya pedido.

TU TAREA:
1. Identificar qué piezas de información de la base serían útiles para responder la pregunta SIN CAMBIAR su naturaleza.
2. Descomponer la pregunta en una lista de subconsultas bien definidas.
3. Para cada subconsulta, indicar:
   - id: un identificador corto sin espacios (ejemplo: "hombres_diabetes").
   - description: para qué sirve esa subconsulta en el razonamiento clínico.
   - question: una pregunta en lenguaje natural que describa EXACTAMENTE qué datos se deben obtener, fiel a lo que pide el colega.

FORMATO DE RESPUESTA (ESTRICTO):
Responde SOLO un JSON válido, sin comentarios, sin explicación adicional, con esta estructura exacta:

{{
  "subqueries": [
    {{
      "id": "string",
      "description": "string",
      "question": "string"
    }}
  ]
}}

Reglas adicionales:
- Incluye al menos una subconsulta.
- No inventes tablas ni campos; usa solo lo que esté implícito en el esquema (patients, diagnoses, clinical_notes).
""".strip()

        try:
            response = LLM.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw = getattr(response, "content", str(response)).strip()

            # Intentamos extraer JSON (primer { ... último })
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                json_str = raw[start:end]
                plan = json.loads(json_str)
            except Exception:
                LOG.warning(f"No se pudo parsear JSON del plan, raw: {raw}")
                raise

            # Validación muy simple
            if not isinstance(plan, dict) or "subqueries" not in plan:
                raise ValueError("Plan sin 'subqueries'")

            if not plan["subqueries"]:
                raise ValueError("Lista de subqueries vacía")

            return plan

        except Exception as e:
            LOG.warning(f"Fallo plan_data_requirements, uso plan trivial: {e}")
            return {
                "subqueries": [
                    {
                        "id": "main",
                        "description": "Consulta principal",
                        "question": query,
                    }
                ]
            }

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

                # Validación con EXPLAIN usando el engine de SQLAlchemy
                db = self.engine
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

    def analyze_results(self, main_query: str, bundles: List[Dict[str, Any]]) -> str:
        """
        Genera una opinión clínica basada en:
        - main_query: pregunta clínica original del médico
        - bundles: lista de dicts con la info de cada subconsulta:
          [
            {
              "id": str,
              "description": str,
              "question": str,
              "sql": str,
              "rows": List[Dict]
            },
            ...
          ]
        """

        if not LLM:
            # Fallback simple si no hay LLM configurado
            return f"Resultados (sin análisis clínico por falta de LLM): {bundles}"

        system_prompt = """
Eres un MÉDICO ESPECIALISTA que asiste a otros médicos interpretando datos ya analizados.

SIEMPRE asume que tu lector ES UN PROFESIONAL DE LA SALUD, NO el paciente.

RECIBES SIEMPRE:
- Una PREGUNTA CLÍNICA formulada por otro médico.
- VARIOS BLOQUES DE DATOS CLÍNICOS agregados (resultados de diferentes subconsultas a la base).
- OPCIONALMENTE, un BLOQUE DE CONTEXTO TÉCNICO que NO debes mencionar.

IMPORTANTE SOBRE EL CONTEXTO TÉCNICO:
- El esquema de la base de datos, las consultas y los nombres de tablas/campos se te dan SOLO para que entiendas mejor qué representan los números.
- NUNCA debes mencionar ni describir:
  - SQL, queries, tablas, columnas, campos, bases de datos, tipos de datos, JSON.
  - Nombres de tablas o columnas.
- Tu respuesta debe ser 100% clínica, como si solo hubieras recibido resúmenes numéricos.

MANEJO DE DATOS INCOMPLETOS:
- Si una subconsulta tiene un resultado vacío (lista de filas vacía), debes interpretarlo como:
  "No hay datos disponibles para esa subconsulta en la cohorte analizada".
- Si una subconsulta contiene un error en lugar de datos, debes tratarla como información NO disponible.
- En ambos casos:
  - NO inventes números.
  - NO infieras prevalencias ni cantidades a partir de subconsultas sin datos.
  - Puedes mencionar, en términos clínicos generales, que la pregunta no puede ser respondida completamente por falta de datos.

TU TAREA:
1. Responder a la PREGUNTA CLÍNICA PRINCIPAL con una OPINIÓN CLÍNICA razonada, basándote en:
   - Los datos numéricos disponibles en las distintas subconsultas.
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

        # Texto “clínico” de los resultados para que el modelo los use
        datos_clinicos_lines = []
        for i, b in enumerate(bundles, start=1):
            rows = b.get("rows") or []
            resumen_resultado: str

            if not rows:
                resumen_resultado = "SIN DATOS: la consulta no devolvió filas."
            elif len(rows) == 1 and isinstance(rows[0], dict) and "error" in rows[0]:
                resumen_resultado = f"SIN DATOS: la consulta produjo un error y no se dispone de información utilizable."
            else:
                resumen_resultado = (
                    f"{len(rows)} fila(s) devueltas. Ejemplo de fila: {rows[0]}"
                    if isinstance(rows[0], dict)
                    else f"{len(rows)} fila(s) devueltas."
                )

            datos_clinicos_lines.append(
                f"- Subconsulta {i} (id='{b.get('id', '')}')\n"
                f"  Descripción: {b.get('description', '')}\n"
                f"  Pregunta de datos: {b.get('question', '')}\n"
                f"  Resumen de resultado: {resumen_resultado}\n"
            )

        datos_clinicos_text = "\n".join(datos_clinicos_lines)

        # Contexto técnico completo (SQL + schema) SOLO para el modelo
        schema_text = _build_schema_text(self.schema) if self.schema else "No schema disponible"
        contexto_tecnico = {
            "schema": self.schema,
            "subqueries": [
                {
                    "id": b.get("id"),
                    "question": b.get("question"),
                    "sql": b.get("sql"),
                    "rows": b.get("rows"),
                }
                for b in bundles
            ],
        }
        contexto_tecnico_json = json.dumps(contexto_tecnico, ensure_ascii=False, indent=2)

        user_prompt = f"""
Pregunta clínica principal del colega:
{main_query}

Datos clínicos agregados (resultado de varias subconsultas, en forma de resumen clínico):
{datos_clinicos_text}

Contexto técnico (SOLO PARA TI, NO MENCIONAR EN LA RESPUESTA):
Schema disponible:
{schema_text}

Detalle técnico de subconsultas (SQL + resultados crudos):
{contexto_tecnico_json}

Redacta tu opinión clínica siguiendo las instrucciones del sistema, respondiendo a la PREGUNTA CLÍNICA PRINCIPAL.
""".strip()

        try:
            response = LLM.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return content.strip()
            return str(response).strip()
        except Exception as e:
            LOG.error(f"Error en analyze_results: {e}")
            return f"Resultados (no se pudo generar análisis clínico): {bundles}"

    def run(self, query: str) -> str:
        # Mostrar schema disponible para debug
        if not self.schema or not self.schema.get("tables"):
            return "Error: No se pudo obtener el schema de la base de datos. Verifica la conexión."

        # 1) Planificar qué datos hacen falta
        plan = self.plan_data_requirements(query)
        subqueries = plan.get("subqueries", [])
        if not subqueries:
            subqueries = [
                {
                    "id": "main",
                    "description": "Consulta principal (fallback)",
                    "question": query,
                }
            ]

        bundles: List[Dict[str, Any]] = []

        # 2) Para cada subconsulta, generamos SQL y lo ejecutamos
        for sq in subqueries:
            sub_id = sq.get("id", "sin_id")
            sub_desc = sq.get("description", "")
            sub_question = sq.get("question", query)

            sql = self.generate_sql(sub_question)
            rows = self.execute_query(sql)

            bundles.append({
                "id": sub_id,
                "description": sub_desc,
                "question": sub_question,
                "sql": sql,
                "rows": rows,
            })

        # 3) Pasar TODOS los resultados al agente clínico
        analysis = self.analyze_results(query, bundles)

        # 4) Info de schema para debug
        schema_info = "\n".join(
            [f"{table}: {', '.join(info['columns'])}" for table, info in self.schema["tables"].items()]
        )

        # 5) Info técnica de subconsultas para debug (opcional)
        debug_blocks = []
        for b in bundles:
            debug_blocks.append(
                f"--- SUBCONSULTA {b.get('id')} ---\n"
                f"Descripción: {b.get('description')}\n"
                f"Pregunta de datos: {b.get('question')}\n"
                f"SQL:\n{b.get('sql')}\n"
                f"RESULTADO:\n{b.get('rows')}\n"
            )
        debug_text = "\n".join(debug_blocks)

        return (
            f"{analysis}\n\n"
            f"--- SCHEMA DISPONIBLE ---\n{schema_info}\n\n"
            f"--- DETALLE DE SUBCONSULTAS (DEBUG) ---\n{debug_text}"
        )


sql_agent = SQLAgent()

def run_sql_agent(query: str) -> str:
    return sql_agent.run(query)
