"""
Agente Juez Optimizado – Versión 2025
Menos propenso a JSON roto, más estable, más rápido y más limpio.
"""

import os
import json
import logging
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from RAG_agent.RAG_agent import create_rag_chain   # tu nueva versión
from sql_agent import run_sql_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("judge")

# ------------------------------
# CONFIG
# ------------------------------

LLM = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gpt-oss:20B"),
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0.0,
)

# ------------------------------
# ESTADO
# ------------------------------

class JudgeState(TypedDict):
    prompt: str
    rag_response: str
    rag_data: list  # Documentos recuperados por RAG
    sql_response: str
    sql_data: list  # Datos/resultados de consultas SQL
    judgment: dict
    final_answer: str


# ============================================================
# 📌 UTILIDADES JSON – Simple, robusto, sin parches kilométricos
# ============================================================

def extract_json(text: str):
    """Extrae el primer bloque JSON válido del texto."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON found")
    return json.loads(text[start:end])


# ============================================================
# 📌 LLAMADAS A AGENTES
# ============================================================

def call_rag(prompt: str):
    chain = create_rag_chain(k=12)  # Balanceado: suficiente para cobertura completa sin exceso
    return chain.invoke(prompt)


def call_sql(prompt: str):
    return run_sql_agent(prompt)


# ============================================================
# 📌 NODOS
# ============================================================

def rag_node(state: JudgeState):
    log.info("→ Ejecutando RAG")
    result = call_rag(state["prompt"])
    # Manejar compatibilidad: si es string, convertir a dict
    if isinstance(result, str):
        return {"rag_response": result, "rag_data": []}
    return {
        "rag_response": result.get("response", result),
        "rag_data": result.get("documents", [])
    }


def sql_node(state: JudgeState):
    log.info("→ Ejecutando SQL")
    result = call_sql(state["prompt"])
    # Manejar compatibilidad: si es string, convertir a dict
    if isinstance(result, str):
        return {"sql_response": result, "sql_data": []}
    return {
        "sql_response": result.get("response", result),
        "sql_data": result.get("data", [])
    }


def judge_node(state: JudgeState):
    log.info("→ Evaluando respuestas")

    prompt = state["prompt"]
    rag = state["rag_response"]
    sql = state["sql_response"]
    # Los datos se incluirán en la salida final, no en el juicio

    judge_prompt = f"""
## 🧑‍⚖️ **Eres un Evaluador Médico Objetivo y Preciso**

Tu tarea es comparar dos respuestas dadas por:

* **Agente RAG** → basado en recuperación semántica desde una base vectorial.
* **Agente SQL** → basado en datos estructurados desde una base relacional.

Debes determinar cuál respuesta es **más correcta**, **más útil clínicamente** y **más fiel al dataset real**.

Tu juicio debe basarse **únicamente** en la base de conocimientos provista abajo.
No inventes datos. No asumas información no presente.

---

## 📋 CONSULTA ORIGINAL

{prompt}

---

## 📝 RESPUESTAS A EVALUAR

### 🔍 RESPUESTA DEL AGENTE RAG
(Basada en recuperación semántica de base vectorial)

{rag}

---

### 💾 RESPUESTA DEL AGENTE SQL
(Basada en consultas estructuradas a base relacional)

{sql}

---

---

# 📘 **BASE DE CONOCIMIENTOS CLÍNICA (RESUMEN COMPLETO)**

## 👤 Pacientes (id → datos)

1. Eliseo Castejón, 39, M
2. Benigno Acedo, 65, M
3. Candelario Carmona, 84, F
4. Marisela Zabala, 38, M
5. Amando Comas, 51, M
6. Macarena Jaén, 74, F
7. Ana Sofía Juárez, 58, F
8. Pancho Lloret, 48, M
9. Tania Alonso, 44, F
10. Fabiana Colomer, 88, F

---

## 📝 Notas clínicas (síntomas + antecedentes)

### Paciente 1 — Eliseo Castejón

Síntomas: fatiga, mareos, tos seca, dolor abdominal
Antecedentes: Asma, Migraña crónica, Hipertensión, Gastritis

### Paciente 2 — Benigno Acedo

Síntomas: dolor abdominal, cefalea, náuseas, tos seca, fiebre
Antecedentes: Lumbalgia, Diabetes 2, Anemia, Gastritis, Arritmia

### Paciente 3 — Candelario Carmona

Síntomas: fiebre, visión borrosa, náuseas
Antecedentes: Insuficiencia renal, Lumbalgia, Asma

### Paciente 4 — Marisela Zabala

Síntomas: disnea, cefalea, dolor torácico, visión borrosa
Antecedentes: Anemia, Arritmia, COVID prolongado, Diabetes 2, Gastritis

### Paciente 5 — Amando Comas

Síntomas: disnea, dolor torácico, fatiga
Antecedentes: Diabetes 2, Asma, Anemia, Gastritis

### Paciente 6 — Macarena Jaén

Síntomas: visión borrosa, náuseas, dolor abdominal
Antecedentes: Arritmia, COVID prolongado

### Paciente 7 — Ana Sofía Juárez

Síntomas: disnea, fatiga, visión borrosa, cefalea, mareos
Antecedentes: Diabetes 2, Gastritis, Anemia, Hipertensión

### Paciente 8 — Pancho Lloret

Síntomas: fiebre, mareos, cefalea
Antecedentes: Anemia, Gastritis, Asma, COVID prolongado

### Paciente 9 — Tania Alonso

Síntomas: dolor abdominal, visión borrosa, mareos, náuseas
Antecedentes: Asma, Anemia, COVID prolongado, Lumbalgia

### Paciente 10 — Fabiana Colomer

Síntomas: mareos, fatiga extrema, náuseas, disnea, dolor abdominal
Antecedentes: Insuficiencia renal, Migraña crónica, Asma, Gastritis

---

## 🧬 Diagnósticos por paciente (registros SQL y vectoriales)

### Paciente 1

Asma, Hipertensión arterial, Gastritis, Anemia, Migraña crónica

### Paciente 2

Asma (dos veces), Migraña crónica, Gastritis (dos veces)

### Paciente 3

Diabetes 2 (dos veces), Arritmia, Lumbalgia, Migraña crónica

### Paciente 4

Gastritis (dos veces), Insuficiencia renal, Diabetes 2, Migraña crónica

### Paciente 5

Anemia (tres veces), Lumbalgia, Migraña crónica

### Paciente 6

Asma, Insuficiencia renal, Lumbalgia (dos veces), Diabetes 2

### Paciente 7

Diabetes 2 (dos veces), Arritmia, Insuficiencia renal, Gastritis

### Paciente 8

Anemia, Migraña crónica, Diabetes 2, Asma, Insuficiencia renal

### Paciente 9

Lumbalgia, Arritmia, Anemia, Asma, COVID prolongado

### Paciente 10

Lumbalgia (dos veces), Diabetes 2, Asma, Arritmia

---

# 🎯 **TU FUNCIÓN COMO JUEZ**

Debes:

1. Comparar la respuesta del RAG y del SQL.
2. Evaluar solo con la base de conocimientos anterior.
3. No rellenar información no documentada.
4. Ser estricto, técnico y médico.
5. Elegir la respuesta más correcta y útil.

---

# 📊 **CRITERIOS DE EVALUACIÓN**

Evalúa ambas respuestas según:

### 🔍 **Precisión factual**

¿Coincide exactamente con los datos clínicos y diagnósticos reales?

### 🎯 **Relevancia**

¿Responde directamente la pregunta?

### 🧩 **Completitud**

¿Incluye toda la información necesaria?

### 🩺 **Utilidad clínica**

¿La respuesta es clara, útil y segura desde el punto de vista médico?

---

# 🧑‍⚖️ **FORMATO DE SALIDA OBLIGATORIO (SOLO JSON)**

Debes responder EXACTAMENTE en este formato:

```json
{{
  "winner": "rag" | "sql" | "tie",
  "rag_score": 0.0,
  "sql_score": 0.0,
  "reason": "explicación breve",
  "rag_pros": [],
  "rag_cons": [],
  "sql_pros": [],
  "sql_cons": []
}}
```
No incluyas texto fuera del JSON.
"""

    resp = LLM.invoke([
        SystemMessage(content="Responde exclusivamente JSON válido."),
        HumanMessage(content=judge_prompt)
    ]).content

    try:
        parsed = extract_json(resp)
    except Exception as e:
        parsed = {
            "winner": "tie",
            "rag_score": 0.5,
            "sql_score": 0.5,
            "reason": f"JSON inválido: {e}",
            "rag_pros": [],
            "rag_cons": [],
            "sql_pros": [],
            "sql_cons": [],
        }

    return {"judgment": parsed}


def synth_node(state: JudgeState):
    log.info("→ Sintetizando respuesta final")

    j = state["judgment"]
    rag = state["rag_response"]
    sql = state["sql_response"]
    rag_data = state.get("rag_data", [])
    sql_data = state.get("sql_data", [])

    # Selección inteligente del ganador
    if j["winner"] == "rag":
        base = rag
        complement = sql
    elif j["winner"] == "sql":
        base = sql
        complement = rag
    else:
        base = rag
        complement = sql

    # Formatear datos del RAG
    rag_data_text = ""
    if rag_data:
        rag_data_text = "\n\n=== DOCUMENTOS RECUPERADOS POR RAG ===\n"
        for i, doc in enumerate(rag_data[:10], 1):  # Limitar a 10 para no saturar
            content_preview = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            rag_data_text += f"\nDocumento {i}:\n"
            rag_data_text += f"  Contenido: {content_preview}...\n"
            rag_data_text += f"  Metadatos: {metadata}\n"
        if len(rag_data) > 10:
            rag_data_text += f"\n... y {len(rag_data) - 10} documentos más\n"
    
    # Formatear datos del SQL
    sql_data_text = ""
    if sql_data:
        sql_data_text = "\n\n=== DATOS DE CONSULTAS SQL ===\n"
        for i, bundle in enumerate(sql_data, 1):
            sql_data_text += f"\nSubconsulta {i} (ID: {bundle.get('id', 'N/A')}):\n"
            sql_data_text += f"  Descripción: {bundle.get('description', 'N/A')}\n"
            sql_data_text += f"  Pregunta: {bundle.get('question', 'N/A')}\n"
            sql_data_text += f"  SQL: {bundle.get('sql', 'N/A')}\n"
            rows = bundle.get('rows', [])
            sql_data_text += f"  Resultados: {len(rows)} fila(s)\n"
            if rows:
                sql_data_text += f"  Ejemplo: {rows[0] if len(rows) > 0 else 'N/A'}\n"

    synthesis_prompt = f"""
Genera una respuesta médica clara y completa usando lo mejor del contenido base,
pero agregando SOLO la información complementaria que NO esté duplicada.

BASE:
{base}

COMPLEMENTO:
{complement}

Reglas:
- No dupliques información
- No inventes nada
- Sé preciso, clínico y conciso
- Al final incluye ambas respuestas originales

Formato:
=== RESPUESTA SINTETIZADA ===
...

=== RAG ORIGINAL ===
{rag}

=== SQL ORIGINAL ===
{sql}
"""

    out = LLM.invoke([
        SystemMessage(content="Eres un asistente médico experto."),
        HumanMessage(content=synthesis_prompt)
    ]).content

    # Agregar los datos al final de la respuesta
    final_output = f"{out}\n\n"
    
    if rag_data_text:
        final_output += rag_data_text
    
    if sql_data_text:
        final_output += sql_data_text

    return {"final_answer": final_output}


# ============================================================
# 📌 GRAFO
# ============================================================

def create_judge_graph():
    g = StateGraph(JudgeState)

    g.add_node("rag", rag_node)
    g.add_node("sql", sql_node)
    g.add_node("judge", judge_node)
    g.add_node("synth", synth_node)

    # Ejecución en paralelo (pseudo, orden secuencial pero simp)
    g.set_entry_point("rag")
    g.add_edge("rag", "sql")
    g.add_edge("sql", "judge")
    g.add_edge("judge", "synth")
    g.add_edge("synth", END)

    return g.compile()


# ============================================================
# 📌 FUNCIÓN PÚBLICA
# ============================================================

def judge_agents(prompt: str):
    graph = create_judge_graph()

    state = graph.invoke({
        "prompt": prompt,
        "rag_response": "",
        "rag_data": [],
        "sql_response": "",
        "sql_data": [],
        "judgment": {},
        "final_answer": ""
    })

    return state
