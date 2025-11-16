# agentic-rag-vs-sql-and-judge
Comparación del desempeño de modelos RAG y Agentes relacionales en la asistencia médica basada en historias clínicas

## Agente Juez

Este proyecto incluye un agente juez implementado con LangChain y LangGraph que:
1. Recibe un prompt
2. Lo pasa a dos agentes diferentes (RAG y SQL)
3. Juzga cuál respuesta es más correcta
4. Genera una respuesta final combinando lo mejor de ambas

### Instalación

1. Crear y activar el entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Asegurarse de que Ollama esté corriendo:
```bash
# Instalar Ollama desde https://ollama.ai
# Descargar un modelo (ejemplo):
ollama pull llama3.1:8b
```

4. (Opcional) Configurar variables de entorno en `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Uso

Ejecutar el agente juez con un prompt:

```bash
python judge.py "¿Cuáles son los síntomas más comunes de la diabetes tipo 2?"
```

O desde Python:

```python
from judge import judge_agents

result = judge_agents("Tu prompt aquí")
print(result['final_response'])
```

### Personalización

Para usar tus propios agentes, modifica las funciones `call_agent_rag()` y `call_agent_sql()` en `judge.py` para que llamen a tus implementaciones reales.

### Configuración de Ollama

El agente juez usa Ollama localmente. Asegúrate de:
- Tener Ollama instalado y corriendo
- Tener al menos un modelo descargado (ej: `ollama pull llama3.1:8b`)
- Configurar `OLLAMA_MODEL` y `OLLAMA_BASE_URL` en `.env` si usas valores diferentes a los predeterminados

### Arquitectura

El agente juez utiliza LangGraph con el siguiente flujo:
1. **call_agent1**: Llama al agente RAG
2. **call_agent2**: Llama al agente SQL
3. **judge**: Evalúa ambas respuestas y determina un ganador
4. **generate_final**: Sintetiza una respuesta final combinando lo mejor de ambas
