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
from langgraph.graph import StateGraph, END

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
    if "diabetes" in prompt_lower or "diabético" in prompt_lower:
        return """Consulta SQL ejecutada: SELECT * FROM condiciones_medicas WHERE nombre LIKE '%diabetes%' AND tipo = 'tipo2'

Resultados de la base de datos:

CONDICIÓN: Diabetes Tipo 2
CÓDIGO_CIE10: E11
PREVALENCIA: 8.5% de la población adulta

SÍNTOMAS (tabla sintomas_condicion):
- Código S01: Poliuria (frecuencia: 85%)
- Código S02: Polidipsia (frecuencia: 80%)
- Código S03: Polifagia (frecuencia: 75%)
- Código S04: Fatiga (frecuencia: 70%)
- Código S05: Visión borrosa (frecuencia: 45%)

CRITERIOS_DIAGNÓSTICO (tabla criterios_diagnostico):
- Glucosa en ayunas ≥ 126 mg/dL (ID: D001)
- HbA1c ≥ 6.5% (ID: D002)
- Glucosa aleatoria ≥ 200 mg/dL con síntomas (ID: D003)

POBLACIÓN_AFECTADA:
- Edad promedio: 55 años
- Sexo: M 52%, F 48%
- Factores de riesgo más comunes: obesidad (78%), sedentarismo (65%), antecedentes familiares (58%)"""
    
    elif "hipertensión" in prompt_lower or "presión arterial" in prompt_lower:
        return """Consulta SQL ejecutada: SELECT * FROM condiciones_medicas WHERE nombre = 'Hipertensión Arterial'

Resultados de la base de datos:

CONDICIÓN: Hipertensión Arterial
CÓDIGO_CIE10: I10
PREVALENCIA: 32% de la población adulta

SÍNTOMAS (tabla sintomas_condicion):
- Mayoría asintomática (90% de casos)
- Código S11: Dolor de cabeza (frecuencia: 15%)
- Código S12: Mareos (frecuencia: 12%)

CRITERIOS_DIAGNÓSTICO (tabla criterios_diagnostico):
- Presión sistólica ≥ 140 mmHg (ID: D101)
- Presión diastólica ≥ 90 mmHg (ID: D102)
- Requiere 2+ mediciones en diferentes visitas

CLASIFICACIÓN (tabla clasificacion_hta):
- Estadio 1: 140-159/90-99 mmHg
- Estadio 2: ≥160/≥100 mmHg
- Crisis hipertensiva: ≥180/≥120 mmHg

POBLACIÓN_AFECTADA:
- Edad promedio: 58 años
- Prevalencia aumenta con edad: 20-30 años (5%), 60+ años (65%)"""
    
    elif "asma" in prompt_lower:
        return """Consulta SQL ejecutada: 
SELECT c.nombre, s.sintoma, s.frecuencia, d.prueba_diagnostica 
FROM condiciones_medicas c
JOIN sintomas_condicion s ON c.id = s.condicion_id
JOIN diagnosticos d ON c.id = d.condicion_id
WHERE c.nombre = 'Asma'

Resultados de la base de datos:

CONDICIÓN: Asma
CÓDIGO_CIE10: J45
PREVALENCIA: 7.7% de la población

SÍNTOMAS (tabla sintomas_condicion):
- Sibilancia: frecuencia 92%
- Disnea: frecuencia 88%
- Opresión torácica: frecuencia 75%
- Tos nocturna: frecuencia 68%

PRUEBAS DIAGNÓSTICAS (tabla diagnosticos):
- Espirometría: sensibilidad 85%
- Prueba broncodilatadora: sensibilidad 78%
- Flujo espiratorio máximo: sensibilidad 72%

TIPOS (tabla tipos_asma):
- Asma alérgica: 60% de casos
- Asma no alérgica: 40% de casos
- Asma inducida por ejercicio: 35% de casos"""
    
    elif "síntoma" in prompt_lower or "sintoma" in prompt_lower:
        return """Consulta SQL ejecutada: SELECT * FROM sintomas WHERE activo = 1 ORDER BY frecuencia DESC

Resultados de la base de datos:

TABLA: sintomas
Total de registros: 1,247 síntomas únicos

SÍNTOMAS MÁS FRECUENTES (TOP 10):
1. Fatiga - frecuencia: 23.5% de consultas
2. Dolor de cabeza - frecuencia: 18.2%
3. Fiebre - frecuencia: 15.8%
4. Tos - frecuencia: 14.3%
5. Dolor abdominal - frecuencia: 12.1%
6. Náuseas - frecuencia: 11.5%
7. Dificultad para respirar - frecuencia: 9.8%
8. Dolor en el pecho - frecuencia: 8.4%
9. Mareos - frecuencia: 7.9%
10. Dolor articular - frecuencia: 6.7%

RELACIONES (tabla sintoma_condicion):
- Cada síntoma puede estar asociado a múltiples condiciones
- Relaciones más comunes: fatiga → 45 condiciones, fiebre → 38 condiciones"""
    
    else:
        return f"""Consulta SQL ejecutada: 
SELECT * FROM consultas_medicas 
WHERE pregunta LIKE '%{prompt[:50]}%' 
ORDER BY fecha_consulta DESC 
LIMIT 5

Resultados de la base de datos:

El agente SQL ha ejecutado una consulta estructurada en la base de datos médica relacional.

ESTRUCTURA DE DATOS:
- Tabla: consultas_medicas
- Registros encontrados: 0 (consulta muy específica)
- Tiempo de ejecución: 0.023 segundos

TABLAS RELACIONADAS DISPONIBLES:
- condiciones_medicas (2,341 registros)
- sintomas (1,247 registros)
- diagnosticos (3,892 registros)
- pacientes (45,231 registros)
- historias_clinicas (128,456 registros)

Para obtener resultados más precisos, se recomienda refinar la consulta con términos más específicos o usar JOINs con tablas relacionadas."""


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
