import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Rutas relativas al archivo RAG_agent.py
_RAG_AGENT_DIR = Path(__file__).parent
DATA_DIR = _RAG_AGENT_DIR / "data"
CHROMA_DIR = _RAG_AGENT_DIR / "chroma_db"
COLLECTION_NAME = "historias_clinicas"

LLM_MODEL = "gemma3:1b"
EMBEDDINGS_MODEL = "nomic-embed-text"


def load_documents():
    #Cargamos todas las historias clínicas desde la carpeta data/
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio {DATA_DIR}. "
            f"Crealo y poné ahí tus historias clínicas."
        )

    docs = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            if path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            elif path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())

    if not docs:
        raise ValueError(
            f"No se encontraron documentos en {DATA_DIR}. "
            f"Agregá historias clínicas simuladas en .txt o .pdf."
        )

    return docs


def split_documents(docs):
    # Hace chunking de los documentos para mejorar el RAG.
    # Ajustá chunk_size y chunk_overlap según tus textos.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store_from_scratch():
    # Crea el índice en Chroma a partir de los documentos en data/.
    # Se persiste en chroma_db/ para reusar entre ejecuciones.
    print("Cargando documentos...")
    docs = load_documents()
    print(f"   -> {len(docs)} documentos encontrados")

    print("Creamos los chunks de los documentos")
    splits = split_documents(docs)
    print(f"   -> {len(splits)} chunks generados")

    print("Creando embeddings")
    
    import os
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST,
    )


    print("Creando vectores")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print("Indice construido y persistido en", CHROMA_DIR)
    return vectordb


def get_vector_store():
    # Si existe un índice persistido en chroma_db/, lo carga.
    # Si no, lo construye desde cero.
    import os
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDINGS_MODEL,
        base_url=OLLAMA_HOST,
    )

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return build_vector_store_from_scratch()

    print("Cargando índice Chroma existente")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectordb


def format_docs(docs):
    """
    Formatea los documentos recuperados para inyectarlos al prompt.
    Le agrega numeración para debugging y evaluación.
    """
    partes = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "desconocido")
        partes.append(
            f"[FRAGMENTO {i} - {source}]\n{d.page_content.strip()}"
        )
    return "\n\n".join(partes)


def create_adaptive_retriever(vectordb, k: int = 4, score_threshold: float = None):
    """
    Crea un retriever adaptativo que puede usar umbral de similitud.
    
    Args:
        vectordb: Base de datos vectorial
        k: Número máximo de documentos a recuperar
        score_threshold: Umbral mínimo de similitud (0.0-1.0). 
                        Chroma usa distancia coseno (0.0 = idéntico, 2.0 = opuesto).
                        Para convertir a similitud: similitud = 1 - (distancia/2)
                        Si se proporciona, filtra documentos con similitud < threshold.
    """
    if score_threshold is None:
        # Comportamiento simple: retornar top k
        return vectordb.as_retriever(search_kwargs={"k": k})
    
    # Wrapper que filtra por umbral de similitud
    def filter_by_similarity(query: str):
        # Obtener más candidatos para tener opciones de filtrar
        docs_with_scores = vectordb.similarity_search_with_score(query, k=k * 3)
        
        # Chroma retorna distancia (menor = más similar)
        # Convertir distancia a similitud: similitud = 1 - (distancia/2)
        # O simplemente usar distancia máxima equivalente
        # Para distancia coseno: 0.0 = idéntico, 2.0 = opuesto
        max_distance = 2.0 - (score_threshold * 2.0)  # Convertir threshold a distancia máxima
        
        filtered = [
            doc for doc, distance in docs_with_scores 
            if distance <= max_distance
        ][:k]
        return filtered
    
    # Crear un retriever personalizado
    from langchain_core.retrievers import BaseRetriever
    
    class ThresholdRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str):
            return filter_by_similarity(query)
        
        async def _aget_relevant_documents(self, query: str):
            return filter_by_similarity(query)
    
    return ThresholdRetriever()


def create_rag_chain(k: int = 4, score_threshold: float = None):
    """
    Crea la cadena RAG
    
    Args:
        k: Número máximo de documentos a recuperar (por defecto 4).
           Si usas score_threshold, este es el máximo que se retornará.
        score_threshold: Umbral mínimo de similitud (0.0-1.0). 
                        Si se proporciona, solo se incluyen documentos con score >= threshold.
                        Esto permite recuperar solo documentos realmente relevantes,
                        independientemente de cuántos sean (hasta k máximo).
                        Si None, se usan los top k documentos sin filtrar por similitud.
                        
    Ejemplos:
        # Recuperar top 4 documentos (comportamiento por defecto)
        create_rag_chain(k=4)
        
        # Recuperar solo documentos con similitud >= 0.7 (máximo 10)
        create_rag_chain(k=10, score_threshold=0.7)
        
        # Recuperar documentos muy similares (>= 0.9), máximo 5
        create_rag_chain(k=5, score_threshold=0.9)
    """
    vectordb = get_vector_store()
    retriever = create_adaptive_retriever(vectordb, k=k, score_threshold=score_threshold)

    import os

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
    )

    system_template = """
Eres un asistente médico que responde exclusivamente en base
a la información de historias clínicas proporcionadas en el CONTEXTO.

Reglas:
- Si la información no está en el contexto, responde claramente que no puedes asegurarlo.
- No inventes diagnósticos ni medicaciones.
- Si la pregunta es ambigua, acláralo en la respuesta.
- Cuando se te pida listar pacientes o casos, asegúrate de revisar TODO el contexto proporcionado y listar TODOS los casos encontrados, sin omitir ninguno.
- Si hay múltiples pacientes mencionados en el contexto, incluye a todos en tu respuesta.

Contexto:
{context}
"""

    human_template = """
Consulta del usuario:
{question}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def chat_loop():
    print("Escribí tu pregunta médica (o 'salir' para terminar).\n")

    rag_chain = create_rag_chain()

    while True:
        try:
            question = input("👤> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Saliendo...")
            break

        if question.lower() in {"salir", "exit", "quit"}:
            print("👋 Listo, nos vemos.")
            break

        if not question:
            continue

        print("🤖> (Pensando...)\n")
        answer = rag_chain.invoke(question)
        print(f"🤖 {answer}\n")


if __name__ == "__main__":
    chat_loop()