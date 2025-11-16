"""
Agente Juez usando LangChain y LangGraph.
Recibe un prompt, lo pasa a dos agentes, juzga sus respuestas y genera una respuesta final.
"""

import os
import sys
import logging
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Agregar el directorio RAG_agent al path para importar el módulo
RAG_AGENT_DIR = Path(__file__).parent / "RAG_agent"
if str(RAG_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_AGENT_DIR))

# Importar funciones del agente RAG
try:
    # Importar desde el archivo RAG_agent.py
    import importlib.util
    rag_agent_path = RAG_AGENT_DIR / "RAG_agent.py"
    if rag_agent_path.exists():
        spec = importlib.util.spec_from_file_location("RAG_agent_module", rag_agent_path)
        rag_agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_agent_module)
        create_rag_chain = rag_agent_module.create_rag_chain
        RAG_AGENT_AVAILABLE = True
        LOG.info("Agente RAG importado exitosamente")
    else:
        raise ImportError(f"No se encontró el archivo RAG_agent.py en {RAG_AGENT_DIR}")
except Exception as e:
    RAG_AGENT_AVAILABLE = False
    LOG.warning(f"No se pudo importar el agente RAG: {e}. Se usará simulación.")
    create_rag_chain = None

# Configurar modelo LLM con Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

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


# Variable global para cachear la cadena RAG
_rag_chain_cache = None

def get_rag_chain():
    """Obtiene o crea la cadena RAG (con cache)"""
    global _rag_chain_cache
    if _rag_chain_cache is None and RAG_AGENT_AVAILABLE:
        try:
            _rag_chain_cache = create_rag_chain()
            LOG.info("Cadena RAG creada exitosamente")
        except Exception as e:
            LOG.error(f"Error al crear la cadena RAG: {e}")
            _rag_chain_cache = None
    return _rag_chain_cache


# Funciones para llamar a los agentes (pueden ser reemplazadas por implementaciones reales)
def call_agent_rag(prompt: str) -> str:
    """
    Llama al agente RAG real si está disponible, sino usa simulación.
    """
    # Intentar usar el agente RAG real
    if RAG_AGENT_AVAILABLE:
        try:
            rag_chain = get_rag_chain()
            if rag_chain is not None:
                LOG.info("Llamando al agente RAG (real)...")
                response = rag_chain.invoke(prompt)
                return response
        except Exception as e:
            LOG.warning(f"Error al usar agente RAG real: {e}. Usando simulación.")
    
    # Fallback a simulación
    LOG.info("Llamando al agente RAG (simulado)...")
    
    # Respuestas simuladas realistas basadas en el tipo de pregunta
    prompt_lower = prompt.lower()
    
    # Simulación de respuestas RAG típicas (más descriptivas, basadas en documentos)
    if "diabetes" in prompt_lower or "diabético" in prompt_lower:
        return """La diabetes tipo 2 es una condición crónica caracterizada por resistencia a la insulina y niveles elevados de glucosa en sangre.

Síntomas más comunes:
- Poliuria (micción frecuente)
- Polidipsia (sed excesiva)
- Polifagia (hambre excesiva)
- Fatiga y debilidad
- Visión borrosa
- Cicatrización lenta de heridas
- Infecciones recurrentes

Diagnóstico:
El diagnóstico se realiza mediante análisis de sangre que miden los niveles de glucosa. Los criterios incluyen:
- Glucosa en ayunas ≥ 126 mg/dL
- Glucosa aleatoria ≥ 200 mg/dL con síntomas
- Hemoglobina glicosilada (HbA1c) ≥ 6.5%
- Prueba de tolerancia a la glucosa oral con glucosa ≥ 200 mg/dL a las 2 horas

La diabetes tipo 2 es más común en adultos mayores de 45 años, aunque está aumentando en personas más jóvenes debido a factores de estilo de vida."""
    
    elif "hipertensión" in prompt_lower or "presión arterial" in prompt_lower:
        return """La hipertensión arterial es una condición médica crónica en la que la presión en las arterias está persistentemente elevada.

Síntomas:
La mayoría de las personas con hipertensión no presentan síntomas. Cuando aparecen, pueden incluir:
- Dolor de cabeza
- Dificultad para respirar
- Mareos
- Dolor en el pecho
- Palpitaciones

Diagnóstico:
Se diagnostica mediante múltiples mediciones de presión arterial. Se considera hipertensión cuando:
- Presión sistólica ≥ 140 mmHg
- Presión diastólica ≥ 90 mmHg

Factores de riesgo incluyen edad avanzada, obesidad, sedentarismo, consumo excesivo de sal y alcohol, y antecedentes familiares."""
    
    elif "asma" in prompt_lower:
        return """El asma es una enfermedad crónica de las vías respiratorias caracterizada por inflamación y estrechamiento de los bronquios.

Síntomas principales:
- Sibilancia (silbido al respirar)
- Dificultad para respirar
- Opresión en el pecho
- Tos, especialmente nocturna o temprano en la mañana

Diagnóstico:
Se realiza mediante:
- Historia clínica y examen físico
- Pruebas de función pulmonar (espirometría)
- Prueba de broncodilatador
- Medición del flujo espiratorio máximo
- Pruebas de alergia para identificar desencadenantes

El asma puede ser desencadenado por alérgenos, ejercicio, infecciones respiratorias, cambios climáticos y ciertos medicamentos."""
    
    elif "síntoma" in prompt_lower or "sintoma" in prompt_lower:
        return """Los síntomas son manifestaciones subjetivas de una enfermedad o condición médica que el paciente experimenta y reporta.

Tipos de síntomas:
- Síntomas generales: fiebre, fatiga, malestar general
- Síntomas específicos: relacionados con sistemas orgánicos particulares
- Síntomas agudos: aparecen rápidamente
- Síntomas crónicos: persisten durante tiempo prolongado

Es importante que los pacientes reporten todos los síntomas a su médico, incluyendo cuándo comenzaron, su intensidad, factores que los empeoran o mejoran, y cualquier patrón temporal.

El diagnóstico médico se basa en la combinación de síntomas reportados por el paciente, signos observados en el examen físico, y resultados de pruebas diagnósticas."""
    
    else:
        return f"""Basado en la información recuperada de documentos médicos y literatura especializada:

El agente RAG ha procesado tu consulta sobre: "{prompt[:100]}"

Respuesta basada en recuperación de información:
Los sistemas RAG (Retrieval-Augmented Generation) recuperan información relevante de bases de conocimiento médicas para proporcionar respuestas precisas. 

En este caso, se han identificado documentos relevantes que sugieren que la consulta requiere un análisis detallado de la información médica disponible. La respuesta se construye combinando fragmentos de información recuperados de múltiples fuentes médicas confiables.

Para una respuesta más específica, sería necesario acceder a la base de conocimiento completa y realizar una búsqueda semántica más profunda en los documentos médicos indexados."""


def call_agent_sql(prompt: str) -> str:
    """
    Simula la llamada al agente SQL.
    En producción, esto llamaría a tu implementación real del agente SQL.
    """
    LOG.info("Llamando al agente SQL (simulado)...")
    
    # Respuestas simuladas realistas basadas en consultas SQL típicas
    prompt_lower = prompt.lower()
    
    # Simulación de respuestas SQL típicas (más estructuradas, basadas en datos de BD)
    return """No hay información disponible en la base de datos para la consulta."""


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
Eres un evaluador experto de sistemas de información médica. Tu trabajo es analizar y comparar respuestas de dos agentes de IA que procesan información de historias clínicas.

CONTEXTO: Estás evaluando respuestas de agentes que procesan datos médicos reales de un sistema de historias clínicas. Esta es una tarea legítima de evaluación de sistemas de información médica.

PROMPT ORIGINAL DEL USUARIO:
{prompt}

RESPUESTA DEL AGENTE 1 (RAG - Recuperación de Información):
{agent1_response}

RESPUESTA DEL AGENTE 2 (SQL - Base de Datos):
{agent2_response}

INSTRUCCIONES:
1. Evalúa cuál respuesta es más precisa, completa y útil para responder al prompt original
2. Considera: precisión de los datos, completitud de la información, relevancia al prompt
3. Asigna un score numérico del 0.0 al 1.0 a cada respuesta
4. Determina un ganador: "agent1", "agent2" o "tie"
5. Proporciona razonamiento detallado

IMPORTANTE: Responde ÚNICAMENTE con un objeto JSON válido. No agregues texto adicional antes o después del JSON.

Formato JSON requerido:
{{
    "winner": "agent1",
    "agent1_score": 0.85,
    "agent2_score": 0.60,
    "reasoning": "La respuesta del agente 1 es más completa porque...",
    "agent1_strengths": ["fortaleza 1", "fortaleza 2"],
    "agent2_strengths": ["fortaleza 1"],
    "agent1_weaknesses": ["debilidad 1"],
    "agent2_weaknesses": ["debilidad 1", "debilidad 2"]
}}
"""
    
    messages = [
        SystemMessage(content="""Eres un evaluador experto de sistemas de información médica. 
Tu tarea es evaluar respuestas de agentes de IA que procesan datos de historias clínicas.
Siempre respondes ÚNICAMENTE con un objeto JSON válido, sin texto adicional.
El JSON debe tener exactamente estos campos: winner, agent1_score, agent2_score, reasoning, agent1_strengths, agent2_strengths, agent1_weaknesses, agent2_weaknesses."""),
        HumanMessage(content=judge_prompt)
    ]
    
    try:
        response = LLM.invoke(messages)
        import json
        import re
        
        def clean_json_string(text: str) -> str:
            """Limpia una cadena para hacerla válida como JSON"""
            # Remover caracteres de control inválidos (excepto \t, \n, \r)
            # Los caracteres de control válidos en JSON son: \t (0x09), \n (0x0A), \r (0x0D)
            cleaned = ""
            for char in text:
                code = ord(char)
                # Permitir caracteres de control válidos y todos los demás caracteres
                if code < 32 and char not in ['\t', '\n', '\r']:
                    # Reemplazar caracteres de control inválidos por espacio
                    cleaned += ' '
                else:
                    cleaned += char
            return cleaned
        
        def repair_json_string(text: str) -> str:
            """Repara JSON escapando saltos de línea dentro de cadenas"""
            result = []
            in_string = False
            i = 0
            
            while i < len(text):
                char = text[i]
                
                # Verificar si el carácter está escapado
                is_escaped = i > 0 and text[i-1] == '\\'
                # Contar escapes consecutivos para determinar si realmente está escapado
                escape_count = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    escape_count += 1
                    j -= 1
                is_escaped = escape_count % 2 == 1
                
                if char == '"' and not is_escaped:
                    # Comilla de inicio/fin de cadena
                    in_string = not in_string
                    result.append(char)
                elif in_string and not is_escaped:
                    # Estamos dentro de una cadena y el carácter no está escapado
                    if char == '\n':
                        # Escapar saltos de línea dentro de cadenas
                        result.append('\\n')
                    elif char == '\r':
                        # Escapar retornos de carro dentro de cadenas
                        result.append('\\r')
                    elif char == '\t':
                        # Escapar tabs dentro de cadenas
                        result.append('\\t')
                    else:
                        result.append(char)
                else:
                    # Fuera de cadena o carácter escapado, copiar tal cual
                    result.append(char)
                
                i += 1
            
            return ''.join(result)
        
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
                # Buscar desde la primera { hasta la última }
                start = response_text.find('{')
                if start != -1:
                    # Contar llaves para encontrar el cierre
                    brace_count = 0
                    end = start
                    for i in range(start, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    if end > start:
                        response_text = response_text[start:end]
        
        response_text = response_text.strip()
        
        # Limpiar caracteres de control inválidos
        response_text = clean_json_string(response_text)
        
        # Reparar JSON escapando saltos de línea dentro de cadenas
        response_text = repair_json_string(response_text)
        
        # Intentar parsear el JSON
        judgment = json.loads(response_text)
        
        # Validar que tenga los campos necesarios
        if "winner" not in judgment:
            judgment["winner"] = "tie"
        if "agent1_score" not in judgment:
            judgment["agent1_score"] = 0.5
        if "agent2_score" not in judgment:
            judgment["agent2_score"] = 0.5
        if "reasoning" not in judgment:
            judgment["reasoning"] = "Juicio completado sin razonamiento detallado"
        if "agent1_strengths" not in judgment:
            judgment["agent1_strengths"] = []
        if "agent2_strengths" not in judgment:
            judgment["agent2_strengths"] = []
        if "agent1_weaknesses" not in judgment:
            judgment["agent1_weaknesses"] = []
        if "agent2_weaknesses" not in judgment:
            judgment["agent2_weaknesses"] = []
        
        LOG.info(f"Juicio completado. Ganador: {judgment.get('winner', 'unknown')}")
        
    except json.JSONDecodeError as e:
        LOG.error(f"Error al parsear JSON del juicio: {e}")
        LOG.error(f"Respuesta recibida (primeros 500 chars): {response.content[:500] if 'response' in locals() else 'N/A'}")
        # Intentar extraer información manualmente del texto
        judgment = {
            "winner": "tie",
            "reasoning": f"Error al parsear JSON: {str(e)}. La respuesta del juez no estaba en formato JSON válido.",
            "agent1_score": 0.5,
            "agent2_score": 0.5,
            "agent1_strengths": [],
            "agent2_strengths": [],
            "agent1_weaknesses": [],
            "agent2_weaknesses": []
        }
    except Exception as e:
        LOG.error(f"Error inesperado al juzgar: {e}")
        LOG.error(f"Respuesta recibida (primeros 500 chars): {response.content[:500] if 'response' in locals() else 'N/A'}")
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
Eres un asistente médico experto que sintetiza información de múltiples fuentes para proporcionar respuestas completas y precisas sobre historias clínicas.

CONTEXTO: Estás procesando información médica de un sistema de historias clínicas. Tu tarea es combinar información de dos agentes diferentes para crear una respuesta final mejorada.

PROMPT ORIGINAL DEL USUARIO:
{prompt}

RESPUESTA DEL AGENTE 1 (RAG - Recuperación de Información):
{agent1_response}

RESPUESTA DEL AGENTE 2 (SQL - Base de Datos):
{agent2_response}

EVALUACIÓN REALIZADA:
- Ganador: {judgment.get('winner', 'tie')}
- Score Agente 1: {judgment.get('agent1_score', 0)}
- Score Agente 2: {judgment.get('agent2_score', 0)}
- Razonamiento: {judgment.get('reasoning', 'N/A')}
- Fortalezas Agente 1: {', '.join(judgment.get('agent1_strengths', [])) if judgment.get('agent1_strengths') else 'N/A'}
- Fortalezas Agente 2: {', '.join(judgment.get('agent2_strengths', [])) if judgment.get('agent2_strengths') else 'N/A'}

INSTRUCCIONES:
1. Analiza ambas respuestas y el juicio realizado
2. Combina lo mejor de ambas respuestas
3. Corrige cualquier error o inconsistencia
4. Crea una respuesta final clara, completa y precisa
5. Mantén un tono profesional y médico
6. Incluye las respuestas originales al final para referencia

Formato de salida requerido:

=== RESPUESTA SINTETIZADA ===
[Tu respuesta sintetizada aquí - debe ser clara, completa y profesional]

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
{agent1_response}

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
{agent2_response}
"""
    
    messages = [
        SystemMessage(content="""Eres un asistente médico experto que procesa información de historias clínicas. 
Tu trabajo es sintetizar información de múltiples fuentes para proporcionar respuestas médicas precisas y completas.
Siempre generas respuestas profesionales, claras y bien estructuradas.
Incluyes las respuestas originales de ambos agentes al final para transparencia."""),
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
    workflow = StateGraph(JudgeState)
    
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
