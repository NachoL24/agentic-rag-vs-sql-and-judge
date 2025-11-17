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
# Descargar los modelos necesarios:
ollama pull llama3.1:8b          # Para el juez
ollama pull qwen2.5:0.5b         # Para el agente RAG
ollama pull nomic-embed-text     # Para los embeddings
```

4. Cargar los datos en la base de datos vectorial:
```bash
# Ejecutar el script para cargar los datos de vector_seed.json
python load_vector_data.py
```

**Nota**: Si necesitas borrar y recrear la base de datos vectorial:
```bash
# Borrar la base de datos existente
python clear_vector_db.py

# Luego recrearla
python load_vector_data.py
```

5. (Opcional) Configurar variables de entorno en `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
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

### Integración del Agente RAG

El agente juez ahora está integrado con el agente RAG real ubicado en `RAG_agent/RAG_agent.py`. El sistema:

1. **Intenta usar el agente RAG real** si está disponible y la base de datos vectorial existe
2. **Usa respuestas simuladas** como fallback si el agente RAG no está disponible

El agente RAG utiliza:
- Base de datos vectorial ChromaDB con datos de `seeds/vector_seed.json`
- Modelo de embeddings: `nomic-embed-text`
- Modelo LLM: `qwen2.5:0.5b`

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
