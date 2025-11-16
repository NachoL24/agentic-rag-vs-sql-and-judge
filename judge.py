"""
Agente Juez usando LangChain y LangGraph.
Recibe un prompt, lo pasa a dos agentes, juzga sus respuestas y genera una respuesta final.
"""

import os
import logging
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import clear, END

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Configurar modelo LLM con Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

try:
    LLM = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )
    LOG.info(f"Ollama configurado: {OLLAMA_MODEL} en {OLLAMA_BASE_URL}")
except Exception as e:
    LOG.warning(f"Error al configurar Ollama: {e}. Asegúrate de que Ollama esté corriendo.")
    LLM = None


# Definir el estado del grafo
class JudgeState(TypedDict):
    """Estado del agente juez"""
    prompt: str
    agent1_response: str
    agent2_response: str
    judgment: dict
    final_response: str
    step: str


# Funciones para llamar a los agentes (pueden ser reemplazadas por implementaciones reales)
def call_agent_rag(prompt: str) -> str:
    """
    Simula la llamada al agente RAG.
    En producción, esto llamaría a tu implementación real del agente RAG.
    """
    LOG.info("Llamando al agente RAG...")
    
    # Si tienes una implementación real, reemplaza esto
    # Por ahora, simulamos una respuesta
    if LLM:
        messages = [
            SystemMessage(content="Eres un agente RAG especializado en recuperación de información médica."),
            HumanMessage(content=prompt)
        ]
        response = LLM.invoke(messages)
        return response.content
    else:
        return f"[RAG] Respuesta simulada para: {prompt[:50]}..."


def call_agent_sql(prompt: str) -> str:
    """
    Simula la llamada al agente SQL.
    En producción, esto llamaría a tu implementación real del agente SQL.
    """
    LOG.info("Llamando al agente SQL...")
    
    # Si tienes una implementación real, reemplaza esto
    # Por ahora, simulamos una respuesta
    if LLM:
        messages = [
            SystemMessage(content="Eres un agente SQL especializado en consultas a bases de datos médicas."),
            HumanMessage(content=prompt)
        ]
        response = LLM.invoke(messages)
        return response.content
    else:
        return f"[SQL] Respuesta simulada para: {prompt[:50]}..."


# Nodos del grafo
def call_agent1_node(state: JudgeState) -> JudgeState:
    """Nodo que llama al primer agente (RAG)"""
    LOG.info("Ejecutando nodo: llamar agente 1 (RAG)")
    prompt = state["prompt"]
    response = call_agent_rag(prompt)
    return {
        "agent1_response": response,
        "step": "agent1_completed"
    }


def call_agent2_node(state: JudgeState) -> JudgeState:
    """Nodo que llama al segundo agente (SQL)"""
    LOG.info("Ejecutando nodo: llamar agente 2 (SQL)")
    prompt = state["prompt"]
    response = call_agent_sql(prompt)
    return {
        "agent2_response": response,
        "step": "agent2_completed"
    }


def judge_responses_node(state: JudgeState) -> JudgeState:
    """Nodo que juzga las respuestas de ambos agentes"""
    LOG.info("Ejecutando nodo: juzgar respuestas")
    
    agent1_response = state["agent1_response"]
    agent2_response = state["agent2_response"]
    prompt = state["prompt"]
    
    if not LLM:
        # Fallback si no hay LLM configurado
        judgment = {
            "winner": "agent1",
            "reasoning": "No hay LLM configurado para juzgar",
            "agent1_score": 0.5,
            "agent2_score": 0.5
        }
        return {
            "judgment": judgment,
            "step": "judgment_completed"
        }
    
    # Prompt para el juez
    judge_prompt = f"""
Eres un juez experto que evalúa la calidad de respuestas de agentes de IA en el contexto médico.

Tu tarea es:
1. Evaluar cuál de las dos respuestas es más correcta, completa y útil segun el prompt original.
2. Asignar un score del 0 al 1 a cada respuesta
3. Determinar un ganador
4. Explicar tu razonamiento

PROMPT ORIGINAL:
{prompt}

RESPUESTA DEL AGENTE 1 (RAG):
{agent1_response}

RESPUESTA DEL AGENTE 2 (SQL):
{agent2_response}

Responde en formato JSON con la siguiente estructura:
{{
    "winner": "agent1" o "agent2" o "tie",
    "agent1_score": <número entre 0 y 1>,
    "agent2_score": <número entre 0 y 1>,
    "reasoning": "<explicación detallada de tu evaluación>",
    "agent1_strengths": ["<fortaleza 1>", "<fortaleza 2>", ...],
    "agent2_strengths": ["<fortaleza 1>", "<fortaleza 2>", ...],
    "agent1_weaknesses": ["<debilidad 1>", "<debilidad 2>", ...],
    "agent2_weaknesses": ["<debilidad 1>", "<debilidad 2>", ...]
}}
"""
    
    messages = [
        SystemMessage(content="Eres un juez experto y objetivo. Siempre respondes en formato JSON válido."),
        HumanMessage(content=judge_prompt)
    ]
    
    try:
        response = LLM.invoke(messages)
        import json
        import re
        # Intentar parsear la respuesta como JSON
        response_text = response.content.strip()
        
        # Limpiar markdown code blocks si existen
        if "```" in response_text:
            # Extraer contenido entre ```json y ```
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Si no hay match, intentar extraer cualquier JSON del texto
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
        
        response_text = response_text.strip()
        
        judgment = json.loads(response_text)
        LOG.info(f"Juicio completado. Ganador: {judgment.get('winner', 'unknown')}")
        
    except Exception as e:
        LOG.error(f"Error al parsear juicio: {e}. Respuesta recibida: {response.content[:200] if 'response' in locals() else 'N/A'}")
        # Fallback
        judgment = {
            "winner": "tie",
            "reasoning": f"Error al juzgar: {str(e)}",
            "agent1_score": 0.5,
            "agent2_score": 0.5,
            "agent1_strengths": [],
            "agent2_strengths": [],
            "agent1_weaknesses": [],
            "agent2_weaknesses": []
        }
    
    return {
        "judgment": judgment,
        "step": "judgment_completed"
    }


def generate_final_response_node(state: JudgeState) -> JudgeState:
    """Nodo que genera la respuesta final combinando ambas respuestas"""
    LOG.info("Ejecutando nodo: generar respuesta final")
    
    prompt = state["prompt"]
    agent1_response = state["agent1_response"]
    agent2_response = state["agent2_response"]
    judgment = state["judgment"]
    
    if not LLM:
        # Fallback si no hay LLM configurado
        winner = judgment.get("winner", "tie")
        if winner == "agent1":
            synthesized = agent1_response
        elif winner == "agent2":
            synthesized = agent2_response
        else:
            synthesized = f"Respuesta combinada basada en ambas respuestas:\n\n{agent1_response}\n\n{agent2_response}"
        
        final_response = f"""=== RESPUESTA SINTETIZADA ===
{synthesized}

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
{agent1_response}

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
{agent2_response}"""
        
        return {
            "final_response": final_response,
            "step": "completed"
        }
    
    # Prompt para generar respuesta final
    synthesis_prompt = f"""
Eres un experto que sintetiza las mejores partes de múltiples respuestas para crear una respuesta final superior.

Tu tarea es:
1. Analizar ambas respuestas y el juicio realizado
2. Combinar lo mejor de ambas respuestas
3. Corregir cualquier error o inconsistencia
4. Crear una respuesta final que sea más completa y precisa que cualquiera de las dos individuales

PROMPT ORIGINAL:
{prompt}

RESPUESTA DEL AGENTE 1 (RAG):
{agent1_response}

RESPUESTA DEL AGENTE 2 (SQL):
{agent2_response}

JUICIO:
- Ganador: {judgment.get('winner', 'tie')}
- Score Agente 1: {judgment.get('agent1_score', 0)}
- Score Agente 2: {judgment.get('agent2_score', 0)}
- Razonamiento: {judgment.get('reasoning', 'N/A')}
- Fortalezas Agente 1: {', '.join(judgment.get('agent1_strengths', []))}
- Fortalezas Agente 2: {', '.join(judgment.get('agent2_strengths', []))}

Genera una respuesta final con el siguiente formato:

=== RESPUESTA SINTETIZADA ===
[Aquí va tu respuesta sintetizada que combine lo mejor de ambas respuestas, sea más completa y precisa, corrija errores y sea clara y bien estructurada]

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
{agent1_response}

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
{agent2_response}
"""
    
    messages = [
        SystemMessage(content="Eres un experto en síntesis de información médica. Generas respuestas claras, precisas y completas. Siempre incluyes las respuestas originales de ambos agentes al final."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    try:
        response = LLM.invoke(messages)
        final_response = response.content
        LOG.info("Respuesta final generada exitosamente")
    except Exception as e:
        LOG.error(f"Error al generar respuesta final: {e}")
        # Fallback: incluir ambas respuestas
        winner = judgment.get("winner", "tie")
        if winner == "agent1":
            synthesized = agent1_response
        elif winner == "agent2":
            synthesized = agent2_response
        else:
            synthesized = f"Respuesta combinada basada en ambas respuestas:\n\n{agent1_response}\n\n{agent2_response}"
        
        final_response = f"""=== RESPUESTA SINTETIZADA ===
{synthesized}

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
{agent1_response}

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
{agent2_response}"""
    
    return {
        "final_response": final_response,
        "step": "completed"
    }


# Construir el grafo
def create_judge_graph():
    """Crea y retorna el grafo del agente juez"""
    
    # Crear el grafo
    workflow = clear(JudgeState)
    
    # Agregar nodos
    workflow.add_node("call_agent1", call_agent1_node)
    workflow.add_node("call_agent2", call_agent2_node)
    workflow.add_node("judge", judge_responses_node)
    workflow.add_node("generate_final", generate_final_response_node)
    
    # Definir el flujo
    # Ambos agentes se ejecutan en paralelo
    workflow.set_entry_point("call_agent1")
    workflow.add_edge("call_agent1", "call_agent2")
    workflow.add_edge("call_agent2", "judge")
    workflow.add_edge("judge", "generate_final")
    workflow.add_edge("generate_final", END)
    
    # Compilar el grafo
    app = workflow.compile()
    return app


# Función principal para ejecutar el juez
def judge_agents(prompt: str) -> dict:
    """
    Función principal que ejecuta el agente juez.
    
    Args:
        prompt: El prompt a evaluar
        
    Returns:
        Dict con las respuestas, juicio y respuesta final
    """
    LOG.info(f"Iniciando proceso de juicio para prompt: {prompt[:100]}...")
    
    # Crear el grafo
    app = create_judge_graph()
    
    # Estado inicial
    initial_state = {
        "prompt": prompt,
        "agent1_response": "",
        "agent2_response": "",
        "judgment": {},
        "final_response": "",
        "step": "started"
    }
    
    # Ejecutar el grafo
    final_state = app.invoke(initial_state)
    
    LOG.info("Proceso de juicio completado")
    
    return final_state


if __name__ == "__main__":
    """
    Ejemplo de uso del agente juez.
    """
    import sys
    
    # Obtener el prompt de los argumentos o usar uno por defecto
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "¿Cuáles son los síntomas más comunes de la diabetes tipo 2 y cómo se diagnostica?"
    
    print(f"\n{'='*60}")
    print("AGENTE JUEZ - Evaluando respuestas de dos agentes")
    print(f"{'='*60}\n")
    print(f"Prompt: {prompt}\n")
    
    # Ejecutar el juez
    result = judge_agents(prompt)
    
    # Mostrar resultados
    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}\n")
    
    print(f"Respuesta Agente 1 (RAG):\n{result['agent1_response']}\n")
    print(f"{'-'*60}\n")
    print(f"Respuesta Agente 2 (SQL):\n{result['agent2_response']}\n")
    print(f"{'-'*60}\n")
    
    judgment = result['judgment']
    print(f"JUICIO:")
    print(f"  Ganador: {judgment.get('winner', 'N/A')}")
    print(f"  Score Agente 1: {judgment.get('agent1_score', 0)}")
    print(f"  Score Agente 2: {judgment.get('agent2_score', 0)}")
    print(f"  Razonamiento: {judgment.get('reasoning', 'N/A')}\n")
    print(f"{'-'*60}\n")
    
    print(f"RESPUESTA FINAL (Sintetizada):\n{result['final_response']}\n")
    print(f"{'='*60}\n")
