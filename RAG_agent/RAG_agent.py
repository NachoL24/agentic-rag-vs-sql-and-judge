"""
Agente RAG optimizado para búsqueda clínica.
Incluye: reranking, threshold adecuado, reformulación mejorada,
manejo de contexto, y mejoras en la recuperación.
"""

import os
import json
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Literal

from dotenv import load_dotenv

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# =============================
# CONFIGURACIONES
# =============================

load_dotenv()

_ROOT = Path(__file__).parent
CHROMA_DIR = _ROOT / "chroma_db"
COLLECTION_NAME = "historias_clinicas"
SEED_FILE = _ROOT.parent / "seeds" / "vector_seed.json"

LLM_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20B")
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SIMILARITY_THRESHOLD = 0.9      # antes: 1.5 (muy flojo)
MAX_CONTEXT_CHARS = 8000        # evita sobrepasar el contexto del modelo
MAX_REFORM_QUERY_LEN = 15       # hace queries más efectivas para vectores

# =============================
# ESTADO DEL AGENTE
# =============================

class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    original_query: str
    retrieved_docs: list[Document]
    context: str
    response: str
    k: int
    search_iterations: int
    needs_more_info: bool
    reformulated_query: str


# =============================
# AGENTE
# =============================

class RAGAgent_Optimized:
    """
    Versión optimizada del agente RAG:
    - Mejor recuperación
    - Reranking
    - Reformulación optimizada para búsqueda vectorial
    - Manejo de contexto
    """

    def __init__(self, k=5, max_iterations=3):
        self.k = k
        self.max_iterations = max_iterations

        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.2)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

        self.vectordb = None
        self.agent_graph = None

        self._build_agent()

    # ---------------------------------------------------------------------------
    # VECTOR STORE
    # ---------------------------------------------------------------------------

    def _get_vector_db(self):
        if self.vectordb is not None:
            return self.vectordb

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            print("⚠️ No hay base vectorial encontrada. Creando desde seed...")
            self.vectordb = self._create_from_seed()
        else:
            self.vectordb = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )

        return self.vectordb

    def _create_from_seed(self):
        if not SEED_FILE.exists():
            raise FileNotFoundError("No existe seed JSON ni base vectorial.")

        print(f"📘 Cargando seed desde {SEED_FILE}")
        data = json.load(open(SEED_FILE, "r", encoding="utf-8"))
        docs = []

        for item in data:
            docs.append(Document(
                page_content=item.get("chunk", ""),
                metadata={
                    "patient_id": item.get("patient_id"),
                    "seccion": item.get("seccion", ""),
                    "tipo": item.get("tipo", "nota"),
                    "fecha": item.get("fecha", None),
                }
            ))

        CHROMA_DIR.mkdir(exist_ok=True)
        return Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
        )

    # ---------------------------------------------------------------------------
    # RERANKING
    # ---------------------------------------------------------------------------

    def _rerank(self, query: str, docs: list[Document]):
        """
        Usa el LLM para ordenar los documentos según relevancia.
        Esto mejora muchísimo la calidad RAG.
        """

        if len(docs) <= 1:
            return docs

        prompt = f"""
Evalúa relevancia respecto a esta consulta:

QUERY:
{query}

Lista de documentos (muestra inicial):
{[doc.page_content[:200] for doc in docs]}

Devuelve SOLO un JSON con índices ordenados (ejemplo: [1,0,2]).
"""

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)]).content
            order = json.loads(resp)
            return [docs[i] for i in order if i < len(docs)]
        except:
            return docs

    # ---------------------------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------------------------

    def _search(self, query: str, k: int):
        db = self._get_vector_db()

        results = db.similarity_search_with_score(query, k=k)

        # Filtrar por threshold más estricto
        filtered = [doc for doc, score in results if score < SIMILARITY_THRESHOLD]

        if not filtered:
            filtered = [doc for doc, _ in results]  # fallback

        # RERANK
        return self._rerank(query, filtered)[:k]

    # ---------------------------------------------------------------------------
    # FORMAT CONTEXT
    # ---------------------------------------------------------------------------

    def _format_docs(self, docs):
        if not docs:
            return "No se encontraron documentos relevantes."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"[Documento {i} | Paciente {doc.metadata.get('patient_id')}]\n"
                f"{doc.page_content.strip()}"
            )

        full = "\n\n".join(parts)
        return full[:MAX_CONTEXT_CHARS]  # evitar contextos gigantes

    # ---------------------------------------------------------------------------
    # NODOS DE LANGGRAPH
    # ---------------------------------------------------------------------------

    def _retrieve(self, state: RAGState):
        query = state.get("reformulated_query") or state["query"]
        prev_docs = state.get("retrieved_docs", [])
        iteration = state["search_iterations"]

        print(f"🔍 Recuperando documentos (iter {iteration}) para: {query}")

        new_docs = self._search(query, self.k)

        # Combinar sin duplicar contenidos
        existing_set = {d.page_content for d in prev_docs}
        unique = [d for d in new_docs if d.page_content not in existing_set]

        state["retrieved_docs"] = prev_docs + unique
        state["context"] = self._format_docs(state["retrieved_docs"])
        return state

    def _evaluate(self, state: RAGState):
        prompt = f"""
Evalúa la relevancia de la información recuperada.

QUERY:
{state['original_query']}

CONTEXTO:
{state['context']}

Responde SOLO JSON:
{{
 "es_relevante": true/false,
 "es_suficiente": true/false,
 "falta_informacion": true/false,
 "informacion_necesaria": "texto",
 "necesita_mas_busqueda": true/false
}}
"""

        try:
            raw = self.llm.invoke([HumanMessage(content=prompt)]).content
            json_str = raw[raw.find("{"): raw.rfind("}")+1]
            ev = json.loads(json_str)
        except:
            ev = {"necesita_mas_busqueda": False}

        state["needs_more_info"] = ev.get("necesita_mas_busqueda", False)

        # límite de iteraciones
        if state["search_iterations"] >= self.max_iterations:
            state["needs_more_info"] = False

        return state

    def _reformulate(self, state: RAGState):
        """
        Reformulación optimizada para búsqueda vectorial: menos lingüística,
        más keywords útiles.
        """

        prompt = f"""
Reformula esta consulta SOLO PARA BÚSQUEDA VECTORIAL.

QUERY ORIGINAL:
{state['original_query']}

CONTEXTO:
{state['context'][:1000]}

Reglas:
- Usa máximo {MAX_REFORM_QUERY_LEN} palabras.
- Usa términos clave del contexto.
- NO hagas preguntas.
- NO agregues lenguaje natural innecesario.
- Solo keywords médicas relevantes.

Devuelve SOLO la frase final, sin comillas.
"""

        try:
            reform = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if reform.startswith(("'", '"')) and reform.endswith(("'", '"')):
                reform = reform[1:-1]

            state["reformulated_query"] = reform
            state["query"] = reform
            state["search_iterations"] += 1

            print(f"🔄 Nueva reformulación: {reform}")

        except:
            state["needs_more_info"] = False

        return state

    def _generate(self, state: RAGState):
        system = """
Eres un asistente clínico experto. 
Responde SOLO con información contenida en el contexto.
No inventes nada. Si no está, dilo explícitamente.
"""

        human = f"""
CONTEXTO:
{state['context']}

PREGUNTA:
{state['original_query']}
"""

        out = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human)
        ])

        state["response"] = out.content
        return state

    # ---------------------------------------------------------------------------
    # Routing
    # ---------------------------------------------------------------------------

    def _route(self, state: RAGState):
        if state["needs_more_info"]:
            return "reformulate"
        return "generate"

    # ---------------------------------------------------------------------------
    # BUILD GRAPH
    # ---------------------------------------------------------------------------

    def _build_agent(self):
        g = StateGraph(RAGState)

        g.add_node("retrieve", self._retrieve)
        g.add_node("evaluate", self._evaluate)
        g.add_node("reformulate", self._reformulate)
        g.add_node("generate", self._generate)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self._route,
                                {"reformulate": "reformulate",
                                 "generate": "generate"})
        g.add_edge("reformulate", "retrieve")
        g.add_edge("generate", END)

        self.agent_graph = g.compile()

    # ---------------------------------------------------------------------------
    # INVOCACIÓN
    # ---------------------------------------------------------------------------

    def invoke(self, query: str):
        init = {
            "messages": [],
            "query": query,
            "original_query": query,
            "retrieved_docs": [],
            "context": "",
            "response": "",
            "k": self.k,
            "search_iterations": 0,
            "needs_more_info": False,
            "reformulated_query": ""
        }
        out = self.agent_graph.invoke(init)
        return out["response"]


# ============================================
# WRAPPER PARA JUDGE.PY
# ============================================

def create_rag_chain(k=4, score_threshold=None):
    agent = RAGAgent_Optimized(k=k)

    class Wrapper:
        def __init__(self, agent):
            self.agent = agent

        def invoke(self, query):
            return self.agent.invoke(query)

    return Wrapper(agent)
