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

from RAG_agent.RAG_agent_optimized import create_rag_chain   # tu nueva versión
from sql_agent import run_sql_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("judge")

# ------------------------------
# CONFIG
# ------------------------------

LLM = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0.0,
)

# ------------------------------
# ESTADO
# ------------------------------

class JudgeState(TypedDict):
    prompt: str
    rag_response: str
    sql_response: str
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
    chain = create_rag_chain(k=4)
    return chain.invoke(prompt)


def call_sql(prompt: str):
    return run_sql_agent(prompt)


# ============================================================
# 📌 NODOS
# ============================================================

def rag_node(state: JudgeState):
    log.info("→ Ejecutando RAG")
    return {"rag_response": call_rag(state["prompt"])}


def sql_node(state: JudgeState):
    log.info("→ Ejecutando SQL")
    return {"sql_response": call_sql(state["prompt"])}


def judge_node(state: JudgeState):
    log.info("→ Evaluando respuestas")

    prompt = state["prompt"]
    rag = state["rag_response"]
    sql = state["sql_response"]

    judge_prompt = f"""
Eres un evaluador médico objetivo. Debes comparar dos respuestas.

PROMPT:
{prompt}

RESPUESTA_RAG:
{rag}

RESPUESTA_SQL:
{sql}

Evalúa:

- Precisión factual
- Relevancia
- Grado de completitud
- Utilidad clínica

Responde **solo JSON**, con esta estructura exacta:

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

    return {"final_answer": out}


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
        "sql_response": "",
        "judgment": {},
        "final_answer": ""
    })

    return state
