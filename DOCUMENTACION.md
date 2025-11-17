# Documentación del Agente Juez

## Índice
1. [Introducción](#introducción)
2. [Arquitectura General](#arquitectura-general)
3. [Componentes Principales](#componentes-principales)
4. [Flujo de Ejecución](#flujo-de-ejecución)
5. [Detalles Técnicos](#detalles-técnicos)
6. [Uso del Sistema](#uso-del-sistema)
7. [Personalización](#personalización)

---

## Introducción

El **Agente Juez** es un sistema implementado con **LangChain** y **LangGraph** que evalúa y sintetiza respuestas de dos agentes diferentes (RAG y SQL) para generar una respuesta final mejorada. El sistema está diseñado para el contexto médico, donde se comparan respuestas de un agente basado en Recuperación Aumentada por Generación (RAG) y otro basado en consultas SQL a bases de datos relacionales.

### Objetivo Principal

El agente juez:
1. Recibe un prompt del usuario
2. Distribuye el prompt a dos agentes especializados (RAG y SQL)
3. Evalúa y compara las respuestas de ambos agentes
4. Genera una respuesta final sintetizada que combina lo mejor de ambas respuestas

---

## Arquitectura General

El sistema utiliza **LangGraph** para crear un grafo de estado que orquesta el flujo completo. La arquitectura se basa en un patrón de pipeline con los siguientes componentes:

```
┌─────────────┐
│   PROMPT    │
│   (Input)   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐      ┌─────────────────┐
│  Agente RAG     │      │  Agente SQL     │
│  (Paralelo)     │      │  (Paralelo)     │
└────────┬────────┘      └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌──────────────────┐
         │  Nodo Juez      │
         │  (Evaluación)   │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │  Síntesis Final  │
         │  (Respuesta)     │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │   RESULTADO      │
         │   (Output)       │
         └──────────────────┘
```

### Tecnologías Utilizadas

- **LangChain**: Framework para construir aplicaciones con LLMs
- **LangGraph**: Biblioteca para crear grafos de estado y workflows complejos
- **Ollama**: Motor de LLM local para ejecutar modelos de lenguaje
- **Python 3.9+**: Lenguaje de programación

---

## Componentes Principales

### 1. Estado del Grafo (`JudgeState`)

El estado del grafo es un `TypedDict` que mantiene toda la información durante la ejecución:

```python
class JudgeState(TypedDict):
    prompt: str              # Prompt original del usuario
    agent1_response: str     # Respuesta del agente RAG
    agent2_response: str     # Respuesta del agente SQL
    judgment: dict           # Resultado de la evaluación
    final_response: str      # Respuesta final sintetizada
    step: str               # Paso actual en el flujo
```

### 2. Agente RAG (`call_agent_rag`)

**Función**: Simula o ejecuta el agente RAG que recupera información de documentos médicos.

**Características**:
- Respuestas más narrativas y descriptivas
- Basadas en recuperación de información de documentos
- Estilo más natural y explicativo

**Implementación actual**: Respuestas simuladas que varían según el tipo de pregunta (diabetes, hipertensión, asma, etc.)

### 3. Agente SQL (`call_agent_sql`)

**Función**: Simula o ejecuta el agente SQL que consulta bases de datos relacionales médicas.

**Características**:
- Respuestas estructuradas con datos tabulares
- Incluye consultas SQL simuladas
- Formato más técnico con códigos, frecuencias y estadísticas

**Implementación actual**: Respuestas simuladas con estructura de base de datos (tablas, JOINs, resultados de consultas)

### 4. Nodo Juez (`judge_responses_node`)

**Función**: Evalúa y compara las respuestas de ambos agentes usando un LLM como juez.

**Proceso**:
1. Recibe ambas respuestas y el prompt original
2. Genera un prompt de evaluación para el LLM
3. El LLM juzga cuál respuesta es mejor
4. Asigna scores (0-1) a cada respuesta
5. Determina un ganador o empate
6. Proporciona razonamiento detallado

**Formato de salida**:
```json
{
    "winner": "agent1" | "agent2" | "tie",
    "agent1_score": 0.0-1.0,
    "agent2_score": 0.0-1.0,
    "reasoning": "Explicación detallada...",
    "agent1_strengths": ["...", "..."],
    "agent2_strengths": ["...", "..."],
    "agent1_weaknesses": ["...", "..."],
    "agent2_weaknesses": ["...", "..."]
}
```

**Manejo de errores**: Si el JSON generado por el LLM es inválido, el sistema:
- Limpia caracteres de control inválidos
- Repara saltos de línea dentro de cadenas JSON
- Usa valores por defecto si el parsing falla

### 5. Nodo de Síntesis (`generate_final_response_node`)

**Función**: Genera una respuesta final combinando lo mejor de ambas respuestas.

**Proceso**:
1. Analiza ambas respuestas y el juicio realizado
2. Combina las mejores partes de cada respuesta
3. Corrige errores o inconsistencias
4. Genera una respuesta más completa y precisa

**Formato de salida**:
```
=== RESPUESTA SINTETIZADA ===
[Respuesta sintetizada que combina lo mejor de ambas]

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
[Respuesta completa del agente RAG]

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
[Respuesta completa del agente SQL]
```

---

## Flujo de Ejecución

### Paso 1: Recepción del Prompt
El usuario proporciona un prompt (pregunta o consulta médica).

### Paso 2: Llamada a Agentes (Paralelo)
El grafo ejecuta dos nodos en secuencia (aunque conceptualmente son paralelos):
- **Nodo 1**: `call_agent1_node` → Llama a `call_agent_rag(prompt)`
- **Nodo 2**: `call_agent2_node` → Llama a `call_agent_sql(prompt)`

Cada agente genera su respuesta independientemente.

### Paso 3: Evaluación (Nodo Juez)
El nodo `judge_responses_node`:
1. Recibe ambas respuestas y el prompt original
2. Construye un prompt de evaluación para el LLM
3. Invoca al LLM (Ollama) para juzgar las respuestas
4. Parsea la respuesta JSON del juez
5. Valida y completa campos faltantes
6. Retorna el juicio estructurado

**Prompt de evaluación incluye**:
- El prompt original
- Respuesta del agente RAG
- Respuesta del agente SQL
- Instrucciones para evaluar y asignar scores

### Paso 4: Síntesis Final
El nodo `generate_final_response_node`:
1. Recibe ambas respuestas y el juicio
2. Construye un prompt de síntesis para el LLM
3. El LLM genera una respuesta final sintetizada
4. Formatea la salida incluyendo:
   - Respuesta sintetizada
   - Respuestas originales de ambos agentes

### Paso 5: Retorno del Resultado
El estado final contiene:
- Respuestas de ambos agentes
- Juicio completo con scores y razonamiento
- Respuesta final sintetizada

---

## Detalles Técnicos

### Configuración de Ollama

El sistema usa Ollama como motor de LLM. Configuración:

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # URL por defecto
OLLAMA_MODEL = "llama3.1:8b"                # Modelo por defecto
```

**Variables de entorno** (opcional en `.env`):
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Manejo de JSON

El sistema incluye funciones robustas para manejar JSON generado por LLMs:

#### 1. `clean_json_string(text)`
- Remueve caracteres de control inválidos
- Mantiene caracteres válidos: `\t`, `\n`, `\r`
- Reemplaza otros caracteres de control por espacios

#### 2. `repair_json_string(text)`
- Escapa saltos de línea dentro de cadenas JSON
- Maneja correctamente comillas escapadas
- Detecta si un carácter está dentro de una cadena o no

#### 3. Extracción de JSON
- Busca bloques de código markdown (```json ... ```)
- Si no encuentra, busca el primer `{` y cuenta llaves hasta el cierre
- Maneja JSON incompleto o mal formateado

### Manejo de Errores

El sistema tiene múltiples capas de manejo de errores:

1. **Error de configuración de Ollama**: Usa valores por defecto o fallback
2. **Error de parsing JSON**: Limpia y repara el JSON antes de parsear
3. **Error de campos faltantes**: Valida y completa campos con valores por defecto
4. **Error general**: Retorna un juicio por defecto (tie, scores 0.5)

### Logging

El sistema usa logging de Python para rastrear la ejecución:

```python
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
```

**Eventos registrados**:
- Inicio de cada nodo
- Llamadas a agentes
- Completación de juicios
- Errores y excepciones

---

## Uso del Sistema

### Instalación

1. **Crear entorno virtual**:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar Ollama**:
```bash
# Asegurarse de que Ollama esté corriendo
ollama pull llama3.1:8b
```

4. **Configurar variables de entorno** (opcional):
```bash
# Crear archivo .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Ejecución desde Línea de Comandos

```bash
python judge.py "¿Cuáles son los síntomas de la diabetes tipo 2?"
```

### Ejecución desde Python

```python
from judge import judge_agents

result = judge_agents("¿Cuáles son los síntomas de la diabetes tipo 2?")

# Acceder a los resultados
print(result['agent1_response'])  # Respuesta RAG
print(result['agent2_response'])   # Respuesta SQL
print(result['judgment'])         # Juicio completo
print(result['final_response'])   # Respuesta final
```

### Ejemplo de Salida

```
============================================================
AGENTE JUEZ - Evaluando respuestas de dos agentes
============================================================

Prompt: ¿Cuáles son los síntomas de la diabetes tipo 2?

============================================================
RESULTADOS
============================================================

Respuesta Agente 1 (RAG):
[Respuesta del agente RAG...]

------------------------------------------------------------

Respuesta Agente 2 (SQL):
[Respuesta del agente SQL...]

------------------------------------------------------------

JUICIO:
  Ganador: agent1
  Score Agente 1: 0.85
  Score Agente 2: 0.6
  Razonamiento: [Explicación detallada...]

------------------------------------------------------------

RESPUESTA FINAL (Sintetizada):
=== RESPUESTA SINTETIZADA ===
[Respuesta sintetizada...]

=== RESPUESTA ORIGINAL DEL AGENTE 1 (RAG) ===
[Respuesta RAG completa...]

=== RESPUESTA ORIGINAL DEL AGENTE 2 (SQL) ===
[Respuesta SQL completa...]

============================================================
```

---

## Personalización

### Reemplazar Agentes Simulados

Para usar agentes reales, modifica las funciones `call_agent_rag` y `call_agent_sql`:

```python
def call_agent_rag(prompt: str) -> str:
    """
    Implementación real del agente RAG
    """
    # Tu código aquí para llamar a tu agente RAG real
    # Ejemplo:
    # response = tu_agente_rag.invoke(prompt)
    # return response
    pass

def call_agent_sql(prompt: str) -> str:
    """
    Implementación real del agente SQL
    """
    # Tu código aquí para llamar a tu agente SQL real
    # Ejemplo:
    # response = tu_agente_sql.query(prompt)
    # return response
    pass
```

### Cambiar el Modelo de Ollama

Edita las variables de entorno o el código:

```python
OLLAMA_MODEL = "llama3.2:13b"  # Modelo más grande
# o
OLLAMA_MODEL = "mistral:7b"    # Modelo diferente
```

### Modificar el Prompt de Evaluación

Edita el prompt en `judge_responses_node`:

```python
judge_prompt = f"""
[Tu prompt personalizado aquí]
"""
```

### Modificar el Formato de Salida

Edita el prompt en `generate_final_response_node` para cambiar el formato de la respuesta final.

### Ejecutar Agentes en Paralelo Real

Actualmente los agentes se ejecutan en secuencia. Para paralelismo real, puedes usar:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def call_agents_parallel(prompt):
    with ThreadPoolExecutor() as executor:
        rag_future = executor.submit(call_agent_rag, prompt)
        sql_future = executor.submit(call_agent_sql, prompt)
        return rag_future.result(), sql_future.result()
```

---

## Estructura del Proyecto

```
agentic-rag-vs-sql-and-judge/
├── judge.py              # Código principal del agente juez
├── requirements.txt      # Dependencias de Python
├── README.md            # Documentación básica
├── DOCUMENTACION.md     # Este documento
├── .env                 # Variables de entorno (opcional)
└── venv/               # Entorno virtual de Python
```

---

## Consideraciones Importantes

### Rendimiento

- **Tiempo de ejecución**: Depende del modelo de Ollama y la complejidad del prompt
- **Modelos más grandes**: Mejor calidad pero más lentos
- **Modelos más pequeños**: Más rápidos pero pueden tener menor calidad

### Limitaciones

1. **Respuestas simuladas**: Los agentes RAG y SQL actualmente devuelven respuestas simuladas
2. **Dependencia de Ollama**: Requiere que Ollama esté corriendo localmente
3. **Parsing JSON**: Aunque es robusto, puede fallar con respuestas muy mal formateadas

### Mejoras Futuras

- [ ] Integración con agentes RAG y SQL reales
- [ ] Caché de respuestas para prompts similares
- [ ] Paralelismo real entre agentes
- [ ] Interfaz web o API REST
- [ ] Métricas de evaluación más detalladas
- [ ] Soporte para múltiples modelos de LLM

---

## Troubleshooting

### Error: "Ollama no está corriendo"
**Solución**: Asegúrate de que Ollama esté instalado y corriendo:
```bash
ollama serve
```

### Error: "Modelo no encontrado"
**Solución**: Descarga el modelo:
```bash
ollama pull llama3.1:8b
```

### Error: "Invalid control character"
**Solución**: Ya está manejado automáticamente. Si persiste, verifica que el modelo esté generando JSON válido.

### Error: "JSON parsing failed"
**Solución**: El sistema usa valores por defecto. Revisa los logs para ver la respuesta del LLM.

---

## Referencias

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Python JSON Documentation](https://docs.python.org/3/library/json.html)

---

## Autor

Sistema desarrollado para comparar el desempeño de modelos RAG y Agentes relacionales en la asistencia médica basada en historias clínicas.

---

**Última actualización**: 2024

